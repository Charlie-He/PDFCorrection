# PDF校正应用

PDF校正应用是一个可以上传PDF文件并自动校正倾斜文本的Web应用程序。该工具特别适用于扫描文档中出现的倾斜文本，提高可读性和OCR准确性。

## 功能特性

- 上传PDF文件进行文本倾斜校正
- 自动检测并校正倾斜文本
- 预览校正后的PDF文档
- 下载校正后的PDF文件

## 技术栈

### 后端
- Java
- Spring Boot
- PDFBox (用于PDF处理)
- Maven (用于依赖管理)

### 前端
- Vue.js
- PDF.js (用于PDF渲染)

## 项目结构

```
PDFCorrection/
├── frontend/           # Vue.js前端应用
│   ├── src/
│   │   ├── components/ # Vue组件
│   │   └── ...         # 其他前端文件
│   └── ...             # 前端配置文件
├── src/                # Java后端源代码
│   └── main/
│       ├── java/       # Java源文件
│       │   └── com/example/pdfcorrection/
│       │       ├── controller/  # REST控制器
│       │       ├── service/     # 业务逻辑
│       │       └── PDFApplication.java # 主应用类
│       └── resources/  # 配置文件
└── pom.xml            # Maven配置
```

## 快速开始

### 前置要求

- Java 8 或更高版本
- Maven 3.6+
- Node.js 12+ (用于前端)
- npm 6+ (用于前端)

### 后端设置

1. 导航到项目根目录
2. 运行以下命令构建项目：
   ```
   mvn clean install
   ```
3. 启动应用：
   ```
   mvn spring-boot:run
   ```

### 前端设置

1. 导航到 `frontend` 目录
2. 安装依赖：
   ```
   npm install
   ```
3. 启动开发服务器：
   ```
   npm run serve
   ```

## 使用说明

1. 在浏览器中打开应用（通常是 http://localhost:8080）
2. 使用上传按钮上传PDF文件
3. 系统会自动检测并校正倾斜文本
4. 预览校正后的PDF
5. 如需要可下载校正后的版本

## API 接口

- `POST /api/pdf/upload` - 上传PDF文件进行校正
- `GET /api/pdf/{id}` - 获取校正后的PDF文件
- `GET /api/pdf/{id}/download` - 下载校正后的PDF文件

## 更新历史
- 2025-10-7: 初始版本发布
- 2025-10-24: 三阶段检测策略：
   快速投影分析：粗略估计（-15°到+15°，步长1°）
   Hough线检测：验证和精确角度识别
   投影方差优化：在±2°范围内精细搜索（步长0.1°）