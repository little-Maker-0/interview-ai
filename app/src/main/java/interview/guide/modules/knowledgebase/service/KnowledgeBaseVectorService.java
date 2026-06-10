package interview.guide.modules.knowledgebase.service;

import interview.guide.common.exception.BusinessException;
import interview.guide.common.exception.ErrorCode;
import interview.guide.modules.knowledgebase.model.KnowledgeBaseEntity;
import interview.guide.modules.knowledgebase.repository.KnowledgeBaseRepository;
import interview.guide.modules.knowledgebase.repository.VectorRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.document.Document;
import org.springframework.ai.transformer.splitter.TextSplitter;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.*;
import java.util.stream.Collectors;

/**
 * 知识库向量存储服务
 * 负责文档分块、向量化和检索
 */
@Slf4j
@Service
public class KnowledgeBaseVectorService {

    /**
     * 阿里云 DashScope Embedding API 批量大小限制
     */
    private static final int MAX_BATCH_SIZE = 10;

    /**
     * 次级切分的 chunk 大小（tokens），用于超长段落的 TokenTextSplitter 切分
     */
    private static final int SUB_CHUNK_SIZE = 600;

    /**
     * 重叠窗口最大回退字符数
     */
    private static final int OVERLAP_MAX_LOOKBACK = 60;

    /**
     * 段落预估 token 阈值，超过此值触发次级切分
     */
    private static final int PARAGRAPH_TOKEN_THRESHOLD = 600;

    /**
     * 英语文本 token 估算比例（约 4 字符 = 1 token）
     */
    private static final double EN_CHARS_PER_TOKEN = 4.0;
    private static final int MIN_CHUNK_TOKENS = 200;

    private final VectorStore vectorStore;
    private final TextSplitter textSplitter;
    private final VectorRepository vectorRepository;
    private final KnowledgeBaseRepository knowledgeBaseRepository;

    public KnowledgeBaseVectorService(VectorStore vectorStore,
                                      VectorRepository vectorRepository,
                                      KnowledgeBaseRepository knowledgeBaseRepository) {
        this.vectorStore = vectorStore;
        this.vectorRepository = vectorRepository;
        this.knowledgeBaseRepository = knowledgeBaseRepository;
        this.textSplitter = TokenTextSplitter.builder()
                .withChunkSize(SUB_CHUNK_SIZE)
                .build();
    }

    /**
     * 将知识库内容向量化并存储
     *
     * @param knowledgeBaseId 知识库ID
     * @param content         知识库文本内容
     */
    @Transactional
    public void vectorizeAndStore(Long knowledgeBaseId, String content) {
        log.info("开始向量化知识库: kbId={}, contentLength={}", knowledgeBaseId, content.length());
        try {
            // 1. 先删除该知识库的旧向量数据
            deleteByKnowledgeBaseId(knowledgeBaseId);

            // 2. 结构优先切分
            List<Document> subChunks = splitByStructure(content, knowledgeBaseId);

            if (subChunks.isEmpty()) {
                log.info("文本无有效内容，跳过向量化: kbId={}", knowledgeBaseId);
                updateChunkCount(knowledgeBaseId, 0);
                return;
            }

            log.info("文本分块完成: {} 个subChunks", subChunks.size());

            // 3. 分批向量化并存储（阿里云 DashScope API 限制 batch size <= 10）
            int totalChunks = subChunks.size();
            int batchCount = (totalChunks + MAX_BATCH_SIZE - 1) / MAX_BATCH_SIZE;
            log.info("开始分批向量化: 总共 {} 个subChunks，分 {} 批处理，每批最多 {} 个",
                    totalChunks, batchCount, MAX_BATCH_SIZE);
            for (int i = 0; i < batchCount; i++) {
                int start = i * MAX_BATCH_SIZE;
                int end = Math.min(start + MAX_BATCH_SIZE, totalChunks);
                List<Document> batch = subChunks.subList(start, end);
                log.debug("处理第 {}/{} 批: subChunks {}-{}", i + 1, batchCount, start + 1, end);
                vectorStore.add(batch);
            }

            // 4. 写入 chunk 数量
            updateChunkCount(knowledgeBaseId, totalChunks);

            log.info("知识库向量化完成: kbId={}, subChunks={}, batches={}",
                    knowledgeBaseId, totalChunks, batchCount);
        } catch (Exception e) {
            log.error("向量化知识库失败: kbId={}, error={}", knowledgeBaseId, e.getMessage(), e);
            throw new BusinessException(ErrorCode.KNOWLEDGE_BASE_VECTORIZATION_FAILED,
                    "向量化知识库失败: " + e.getMessage());
        }
    }

    /**
     * 结构优先的文档分块
     * <p>策略：
     * <ol>
     *   <li>按段落（\n\n）拆分，过滤纯噪声段落（纯 URL、纯数字等）</li>
     *   <li>短段落累积合并，达到 MIN_CHUNK_TOKENS 后作为一个 chunk</li>
     *   <li>超长段落丢给 TokenTextSplitter 做次级切分，并对子块加重叠</li>
     *   <li>每个 chunk 标注 source、chunk_index、content_type 等元数据</li>
     * </ol>
     */
    List<Document> splitByStructure(String content, Long knowledgeBaseId) {
        if (content == null || content.isBlank()) {
            return List.of();
        }

        // 获取知识库元数据
        KnowledgeBaseEntity entity = knowledgeBaseRepository.findById(knowledgeBaseId).get();
        String source = entity.getName();
        String contentType = entity.getContentType();

        // 按段落拆分
        String[] paragraphs = content.split("\n\n");
        List<Document> subChunks = new ArrayList<>();
        int chunkIndex = 0;

        // 短段落聚合缓冲区
        StringBuilder mergeBuffer = new StringBuilder();
        int mergeBufferTokens = 0;

        for (String paragraph : paragraphs) {
            String trimmed = paragraph.trim();
            if (trimmed.isEmpty()) {
                continue;
            }

            // 过滤纯噪声段落
            if (isNoiseParagraph(trimmed)) {
                continue;
            }

            int estimatedTokens = estimateTokens(trimmed);

            if (estimatedTokens > PARAGRAPH_TOKEN_THRESHOLD) {
                // 超长段落：先 flush 累积的短段落，再处理当前超长段落
                flushMergeBuffer(mergeBuffer, chunkIndex, knowledgeBaseId, source, contentType, subChunks);
                if (mergeBuffer.length() > 0) {
                    chunkIndex++;
                    mergeBuffer.setLength(0);
                    mergeBufferTokens = 0;
                }

                List<Document> splitResult = textSplitter.apply(List.of(new Document(trimmed)));
                for (Document sub : splitResult) {
                    setChunkMetadata(sub, knowledgeBaseId, source, contentType, chunkIndex);
                    subChunks.add(sub);
                    chunkIndex++;
                }
            } else {
                // 短段落：累积到缓冲区
                if (mergeBuffer.length() > 0) {
                    mergeBuffer.append("\n\n");
                }
                mergeBuffer.append(trimmed);
                mergeBufferTokens += estimatedTokens;

                // 缓冲区达到阈值，生成一个 chunk
                if (mergeBufferTokens >= MIN_CHUNK_TOKENS) {
                    Document doc = new Document(mergeBuffer.toString());
                    setChunkMetadata(doc, knowledgeBaseId, source, contentType, chunkIndex);
                    subChunks.add(doc);
                    chunkIndex++;
                    mergeBuffer.setLength(0);
                    mergeBufferTokens = 0;
                }
            }
        }

        // flush 最后的残留短段落
        if (mergeBuffer.length() > 0) {
            Document doc = new Document(mergeBuffer.toString());
            setChunkMetadata(doc, knowledgeBaseId, source, contentType, chunkIndex);
            subChunks.add(doc);
        }

        // 全局重叠：所有相邻 chunk 之间加重叠，不仅仅是同段落子块
        applyOverlap(subChunks);

        return subChunks;
    }

    /**
     * 将累积的短段落刷新为一个 chunk
     */
    private void flushMergeBuffer(StringBuilder buffer, int index,
                                  Long kbId, String source, String contentType,
                                  List<Document> subChunks) {
        if (buffer.length() == 0) {
            return;
        }
        Document doc = new Document(buffer.toString());
        setChunkMetadata(doc, kbId, source, contentType, index);
        subChunks.add(doc);
    }

    /**
     * 判断段落是否为噪声（纯 URL、纯数字、无实质语义内容）
     */
    private boolean isNoiseParagraph(String text) {
        if (text.isBlank()) {
            return true;
        }

        // 去除空白后的纯文本
        String stripped = text.replaceAll("\\s+", "");

        // 空内容
        if (stripped.isEmpty()) {
            return true;
        }

        // 纯 URL（以 http/https/ftp 开头且占比高）
        if (stripped.startsWith("http://") || stripped.startsWith("https://")
                || stripped.startsWith("ftp://") || stripped.startsWith("www.")) {
            int alphaCount = 0;
            for (char c : stripped.toCharArray()) {
                if (Character.isLetter(c) || Character.UnicodeScript.of(c) == Character.UnicodeScript.HAN) {
                    alphaCount++;
                }
            }
            if (alphaCount < 3) {
                return true;
            }
            // URL 中字母占比 < 40%（被符号和数字主导），判定为纯链接噪声
            if (alphaCount < stripped.length() * 0.4) {
                return true;
            }
        }
        return false;
    }

    /**
     * 对所有相邻 chunk 添加尾句重叠
     * <p>取前一个 chunk 的最后一个完整句子，追加到后一个 chunk 的开头，
     * 确保跨 chunk 边界和跨段落的语义连贯性。
     */
    private void applyOverlap(List<Document> subChunks) {
        if (subChunks.size() < 2) {
            return;
        }

        for (int i = 1; i < subChunks.size(); i++) {
            String prevContent = subChunks.get(i - 1).getText();
            String currContent = subChunks.get(i).getText();

            String tailSentence = extractTailSentence(prevContent);
            if (tailSentence.isEmpty()) {
                continue;
            }

            subChunks.set(i, subChunks.get(i).mutate()
                    .text("[上文衔接: " + tailSentence + "] " + currContent)
                    .build());
        }
    }

    /**
     * 从文本末尾提取最后一个完整句子
     * <p>从尾部向前扫描，找到最近的句子结束符（。！？），最多回退 OVERLAP_MAX_LOOKBACK 字符。
     * 如果未找到句子结束符，回退到换行符；都没有则取全部。
     */
    private String extractTailSentence(String text) {
        if (text == null || text.isBlank()) {
            return "";
        }

        int end = text.length() - 1;
        // 跳过尾部空白
        while (end >= 0 && Character.isWhitespace(text.charAt(end))) {
            end--;
        }

        int searchStart = Math.max(0, end - OVERLAP_MAX_LOOKBACK);

        for (int i = end; i >= searchStart; i--) {
            char c = text.charAt(i);
            if (c == '。' || c == '！' || c == '？' || c == '\n') {
                // 找到句子分隔符，但不要从此处开始（避免以句号开头），
                // 从前一个字符继续往前找到一个合适的起点
                if (c == '\n') {
                    return text.substring(i + 1, end + 1).trim();
                }
                // 从分隔符前面找上一个分隔符或搜索起点
                int sentenceStart = i + 1;
                // 如果上一个分隔符很近（逗号、分号等），多取一个分句
                for (int j = i - 1; j >= searchStart; j--) {
                    char prev = text.charAt(j);
                    if (prev == '。' || prev == '！' || prev == '？' || prev == '\n') {
                        sentenceStart = j + 1;
                        break;
                    }
                    if (prev == '，' || prev == '；') {
                        sentenceStart = j; // 从逗号/分号开始，取完整分句
                        break;
                    }
                }
                return text.substring(sentenceStart, end + 1).trim();
            }
        }

        // 未找到句子分隔符：从 searchStart 处取
        return text.substring(searchStart, end + 1).trim();
    }

    /**
     * 估算文本的 token 数
     * <p>中文约 1 char ≈ 1 token，英文约 4 chars ≈ 1 token。
     * 这是一阶近似，足够用于段落阈值判断。
     */
    private int estimateTokens(String text) {
        if (text == null || text.isEmpty()) {
            return 0;
        }
        int chineseChars = 0;
        int otherChars = 0;
        for (char c : text.toCharArray()) {
            if (Character.UnicodeScript.of(c) == Character.UnicodeScript.HAN) {
                chineseChars++;
            } else if (!Character.isWhitespace(c)) {
                otherChars++;
            }
        }
        return chineseChars + (int) Math.ceil(otherChars / EN_CHARS_PER_TOKEN);
    }

    /**
     * 为 chunk 设置元数据
     */
    private void setChunkMetadata(Document chunk, Long knowledgeBaseId,
                                  String source, String contentType, int chunkIndex) {
        chunk.getMetadata().put("kb_id", knowledgeBaseId.toString());
        chunk.getMetadata().put("source", source);
        chunk.getMetadata().put("chunk_index", chunkIndex);
        chunk.getMetadata().put("content_type", contentType);
    }

    /**
     * 更新知识库的 chunk 数量
     */
    private void updateChunkCount(Long kbId, int count) {
        try {
            knowledgeBaseRepository.findById(kbId).ifPresent(kb -> {
                kb.setChunkCount(count);
                knowledgeBaseRepository.save(kb);
            });
        } catch (Exception e) {
            log.warn("更新chunkCount失败: kbId={}, count={}", kbId, count, e);
        }
    }

    /**
     * 滑动窗口扩展：对检索结果按文档原始顺序补全相邻 chunk
     * <p>检索命中 chunk_i 后，从 pgvector 拉取 chunk_{i-window} 到 chunk_{i+window}
     * 的邻居，按 chunk_index 排序后返回。多个命中窗口重叠时自动去重。
     *
     * @param retrievedChunks 向量检索命中的 chunk 列表
     * @param windowSize      窗口半径（向两侧各取 windowSize 个 chunk）
     * @return 扩展并去重后的 chunk 列表，按 kb_id + chunk_index 有序
     */
    public List<Document> expandWindow(List<Document> retrievedChunks, int windowSize) {
        if (retrievedChunks == null || retrievedChunks.isEmpty() || windowSize <= 0) {
            return retrievedChunks != null ? retrievedChunks : List.of();
        }

        // 按 kb_id 分组，同 kb 内的命中合并区间后批量查询
        Map<String, List<Integer>> hitsByKb = new LinkedHashMap<>();
        for (Document doc : retrievedChunks) {
            Object kbIdObj = doc.getMetadata().get("kb_id");
            Object idxObj = doc.getMetadata().get("chunk_index");
            if (kbIdObj == null || idxObj == null) {
                continue;
            }
            hitsByKb.computeIfAbsent(kbIdObj.toString(), k -> new ArrayList<>())
                    .add(toInt(idxObj));
        }

        if (hitsByKb.isEmpty()) {
            return retrievedChunks;
        }

        // 收集扩展结果，按 (kb_id, chunk_index) 去重
        Set<String> seen = new LinkedHashSet<>();
        List<Document> expanded = new ArrayList<>();

        for (var entry : hitsByKb.entrySet()) {
            String kbId = entry.getKey();
            List<Integer> hitIndices = entry.getValue();

            // 合并重叠区间：相邻命中对应的窗口如果重叠，合并为一次查询
            List<int[]> mergedRanges = mergeOverlappingRanges(hitIndices, windowSize);

            for (int[] range : mergedRanges) {
                List<Document> neighbors = vectorRepository.findNeighborDocuments(
                        kbId, range[0], range[1]);
                for (Document doc : neighbors) {
                    String key = kbId + ":" + doc.getMetadata().get("chunk_index");
                    if (seen.add(key)) {
                        expanded.add(doc);
                    }
                }
            }
        }

        log.debug("滑动窗口扩展: {} 个命中 → {} 个chunks（窗口半径={})",
                retrievedChunks.size(), expanded.size(), windowSize);
        return expanded;
    }

    /**
     * 合并重叠的索引区间，避免重复查询数据库
     */
    private List<int[]> mergeOverlappingRanges(List<Integer> hitIndices, int windowSize) {
        if (hitIndices.isEmpty()) {
            return List.of();
        }

        List<int[]> ranges = hitIndices.stream()
                .sorted()
                .map(i -> new int[]{Math.max(0, i - windowSize), i + windowSize})
                .collect(Collectors.toList());

        List<int[]> merged = new ArrayList<>();
        int[] current = ranges.get(0);
        for (int i = 1; i < ranges.size(); i++) {
            int[] next = ranges.get(i);
            if (next[0] <= current[1] + 1) {
                // 重叠或相邻，合并
                current[1] = Math.max(current[1], next[1]);
            } else {
                merged.add(current);
                current = next;
            }
        }
        merged.add(current);
        return merged;
    }

    private static int toInt(Object obj) {
        if (obj instanceof Integer i) {
            return i;
        }
        if (obj instanceof Number n) {
            return n.intValue();
        }
        try {
            return Integer.parseInt(obj.toString());
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    /**
     * 基于多个知识库进行相似度搜索
     *
     * @param query            查询文本
     * @param knowledgeBaseIds 知识库ID列表（如果为空则搜索所有）
     * @param topK             返回top K个结果
     * @return 相关文档列表
     */
    public List<Document> similaritySearch(String query, List<Long> knowledgeBaseIds, int topK, double minScore) {
        log.info("向量相似度搜索: query={}, kbIds={}, topK={}, minScore={}",
                query, knowledgeBaseIds, topK, minScore);

        try {
            SearchRequest.Builder builder = SearchRequest.builder()
                    .query(query)
                    .topK(Math.max(topK, 1));

            if (minScore > 0) {
                builder.similarityThreshold(minScore);
            }

            if (knowledgeBaseIds != null && !knowledgeBaseIds.isEmpty()) {
                builder.filterExpression(buildKbFilterExpression(knowledgeBaseIds));
            }

            List<Document> results = vectorStore.similaritySearch(builder.build());
            if (results == null) {
                return List.of();
            }

            // Apply topK limiting in case VectorStore returns more than requested
            List<Document> limitedResults = results.stream()
                    .limit(topK)
                    .collect(Collectors.toList());

            log.info("搜索完成: 找到 {} 个相关文档", limitedResults.size());
            return limitedResults;

        } catch (Exception e) {
            log.warn("向量搜索前置过滤失败，回退到本地过滤: {}", e.getMessage());
            return similaritySearchFallback(query, knowledgeBaseIds, topK, minScore);
        }
    }

    private List<Document> similaritySearchFallback(String query, List<Long> knowledgeBaseIds, int topK, double minScore) {
        try {
            // 回退检索仍保留 topK/minScore，避免兜底路径引入过多弱相关命中
            SearchRequest.Builder builder = SearchRequest.builder()
                    .query(query)
                    .topK(Math.max(topK * 3, topK));
            if (minScore > 0) {
                builder.similarityThreshold(minScore);
            }

            List<Document> allResults = vectorStore.similaritySearch(builder.build());
            if (allResults == null || allResults.isEmpty()) {
                return List.of();
            }

            if (knowledgeBaseIds != null && !knowledgeBaseIds.isEmpty()) {
                allResults = allResults.stream()
                        .filter(doc -> isDocInKnowledgeBases(doc, knowledgeBaseIds))
                        .collect(Collectors.toList());
            }

            List<Document> results = allResults.stream()
                    .limit(topK)
                    .collect(Collectors.toList());

            log.info("回退检索完成: 找到 {} 个相关文档", results.size());
            return results;
        } catch (Exception e) {
            log.error("向量搜索失败: {}", e.getMessage(), e);
            throw new BusinessException(ErrorCode.KNOWLEDGE_BASE_QUERY_FAILED,
                    "向量搜索失败: " + e.getMessage());
        }
    }

    private boolean isDocInKnowledgeBases(Document doc, List<Long> knowledgeBaseIds) {
        Object kbId = doc.getMetadata().get("kb_id");
        if (kbId == null) {
            return false;
        }
        try {
            Long kbIdLong = kbId instanceof Long
                    ? (Long) kbId
                    : Long.parseLong(kbId.toString());
            return knowledgeBaseIds.contains(kbIdLong);
        } catch (NumberFormatException e) {
            return false;
        }
    }

    private String buildKbFilterExpression(List<Long> knowledgeBaseIds) {
        String values = knowledgeBaseIds.stream()
                .filter(Objects::nonNull)
                .map(String::valueOf)
                .map(id -> "'" + id + "'")
                .collect(Collectors.joining(", "));
        return "kb_id in [" + values + "]";
    }

    /**
     * 删除指定知识库的所有向量数据
     * 委托给 VectorRepository 处理
     *
     * @param knowledgeBaseId 知识库ID
     */
    @Transactional(rollbackFor = Exception.class)
    public void deleteByKnowledgeBaseId(Long knowledgeBaseId) {
        try {
            vectorRepository.deleteByKnowledgeBaseId(knowledgeBaseId);
        } catch (Exception e) {
            log.error("删除向量数据失败: kbId={}, error={}", knowledgeBaseId, e.getMessage(), e);
        }
    }
}
