package interview.guide.infrastructure.file;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoSettings;

import java.lang.reflect.Method;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * FileStorageService 单元测试
 */
@DisplayName("文件存储服务测试")
@MockitoSettings
class FileStorageServiceTest {

    @Mock
    private software.amazon.awssdk.services.s3.S3Client s3Client;

    @Mock
    private interview.guide.common.config.StorageConfigProperties storageConfig;

    private FileStorageService fileStorageService;

    @BeforeEach
    void setUp() {
        fileStorageService = new FileStorageService(s3Client, storageConfig);
    }

    @Nested
    @DisplayName("convertToPinyin() 测试")
    class ConvertToPinyinTests {

        private Method convertToPinyin;

        @BeforeEach
        void setUp() throws Exception {
            convertToPinyin = FileStorageService.class.getDeclaredMethod("convertToPinyin", String.class);
            convertToPinyin.setAccessible(true);
        }

        private String invokeConvertToPinyin(String input) throws Exception {
            return (String) convertToPinyin.invoke(fileStorageService, input);
        }

        @Test
        @DisplayName("纯汉字转换为大驼峰拼音")
        void testChineseCharacters() throws Exception {
            assertEquals("ZhangSan", invokeConvertToPinyin("张三"));
            assertEquals("LiSi", invokeConvertToPinyin("李四"));
            assertEquals("WangWu", invokeConvertToPinyin("王五"));
        }

        @Test
        @DisplayName("汉字+英文字母混合")
        void testChineseWithEnglish() throws Exception {
            assertEquals("ZhangSanresume", invokeConvertToPinyin("张三resume"));
            assertEquals("resumeLiSi", invokeConvertToPinyin("resume李四"));
        }

        @Test
        @DisplayName("汉字+数字混合")
        void testChineseWithNumbers() throws Exception {
            assertEquals("ZhangSan2024", invokeConvertToPinyin("张三2024"));
            assertEquals("123LiSi", invokeConvertToPinyin("123李四"));
        }

        @Test
        @DisplayName("纯英文/数字保持不变")
        void testNonChineseUnchanged() throws Exception {
            assertEquals("resume.pdf", invokeConvertToPinyin("resume.pdf"));
            assertEquals("abc-123_test.txt", invokeConvertToPinyin("abc-123_test.txt"));
        }

        @Test
        @DisplayName("中文标点/特殊字符替换为下划线")
        void testChinesePunctuation() throws Exception {
            assertEquals("ZhangSanJianLi", invokeConvertToPinyin("张三简历"));
            assertEquals("Zhang_San", invokeConvertToPinyin("张 三"));
        }

        @Test
        @DisplayName("多音字取第一个读音")
        void testPolyphoneFirstReading() throws Exception {
            String result = invokeConvertToPinyin("长江");
            assertTrue(result.startsWith("Chang") || result.startsWith("Zhang"));
        }

        @Test
        @DisplayName("空字符串返回空")
        void testEmptyInput() throws Exception {
            assertEquals("", invokeConvertToPinyin(""));
        }
    }

    @Nested
    @DisplayName("sanitizeFilename() 测试（间接验证拼音）")
    class SanitizeFilenameTests {

        private Method sanitizeFilename;

        @BeforeEach
        void setUp() throws Exception {
            sanitizeFilename = FileStorageService.class.getDeclaredMethod("sanitizeFilename", String.class);
            sanitizeFilename.setAccessible(true);
        }

        private String invokeSanitize(String input) throws Exception {
            return (String) sanitizeFilename.invoke(fileStorageService, input);
        }

        @Test
        @DisplayName("null 或空返回 unknown")
        void testNullOrDefault() throws Exception {
            assertEquals("unknown", invokeSanitize(null));
            assertEquals("unknown", invokeSanitize(""));
        }

        @Test
        @DisplayName("中文文件名转为拼音")
        void testChineseFilename() throws Exception {
            String result = invokeSanitize("张三的简历.pdf");
            assertTrue(result.contains("Zhang"));
            assertTrue(result.contains("JianLi"));
            assertTrue(result.endsWith(".pdf"));
        }
    }
}
