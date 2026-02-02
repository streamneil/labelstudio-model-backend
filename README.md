# Label Studio ML Backend for Kimi (Moonshot)

生产级 Label Studio ML Backend，通过 Model Connector 调用 Kimi 大模型进行自动预标注。

## ✨ 核心特性

| 特性 | 说明 |
|-----|------|
| **Label Studio 兼容** | 完全符合 ML Backend 接口规范 |
| **连接池** | HTTP/2 + 连接复用 |
| **并发控制** | Semaphore 限制并发 API 调用 |
| **批量容错** | 单个任务失败不影响整个批次 |
| **结构化日志** | JSON 格式，便于日志收集分析 |
| **请求追踪** | 每个请求唯一 ID |

## 🚀 快速开始

### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 MOONSHOT_API_KEY
```

### 2. 构建并启动

```bash
docker-compose up -d --build
```

### 3. 验证服务

```bash
# 健康检查
curl http://localhost:8751/health
# {"status":"UP","model":"kimi-k2-0905-Preview"}

# 查看日志
docker-compose logs -f kimi-ml-backend
```

## 🔌 Label Studio 连接配置

1. 打开 Label Studio 项目 → **Settings** → **Model**
2. 点击 **Connect Model**
3. 填写：

| 字段 | 值 |
|-----|-----|
| **Backend URL** | `http://your-server-ip:8751` |
| **Display Name** | `Kimi Auto Labeling` |
| **Description** | `Moonshot Kimi K2 自动标注` |

### 标签配置示例

确保 Label Studio 的标签配置与 Backend 返回格式匹配：

```xml
<View>
  <Text name="text" value="$text"/>
  <TextArea name="label" toName="text" 
            placeholder="模型生成的标注内容..."
            editable="true"/>
</View>
```

**关键对应关系**：
- `Text name="text"` ↔ Backend `to_name: "text"`
- `TextArea name="label"` ↔ Backend `from_name: "label"`

## 🧪 接口测试

```bash
# 测试预测接口
curl -X POST http://localhost:8751/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tasks": [
      {
        "id": 1,
        "data": {"text": "这是一段需要标注的文本内容"}
      }
    ]
  }'
```

响应示例：
```json
[
  {
    "result": [
      {
        "from_name": "label",
        "to_name": "text",
        "type": "textarea",
        "value": {
          "text": ["模型生成的标注内容"]
        }
      }
    ],
    "model_version": "kimi-k2-0905-Preview",
    "score": null
  }
]
```

## ⚙️ 环境变量

| 变量名 | 必填 | 默认值 | 说明 |
|--------|------|--------|------|
| `MOONSHOT_API_KEY` | ✅ | - | Moonshot API 密钥 |
| `MOONSHOT_MODEL` | ❌ | kimi-k2-0905-Preview | 模型名称 |
| `WORKER_COUNT` | ❌ | 4 | Gunicorn worker 数 |
| `MAX_CONCURRENT` | ❌ | 10 | 单 worker 最大并发 |
| `KIMI_API_TIMEOUT` | ❌ | 120 | API 超时（秒） |
| `LOG_LEVEL` | ❌ | INFO | 日志级别 |
| `BIND_PORT` | ❌ | 8751 | 服务端口 |

## 📁 项目结构

```
.
├── main.py              # FastAPI 应用
├── requirements.txt     # Python 依赖
├── Dockerfile           # Docker 镜像构建
├── docker-compose.yml   # Docker Compose 配置
├── .env.example         # 环境变量示例
└── README.md            # 本文档
```

## 🛠️ 常用命令

```bash
# 启动
docker-compose up -d

# 停止
docker-compose down

# 重启
docker-compose restart

# 查看日志
docker-compose logs -f

# 重新构建
docker-compose up -d --build
```

## 🔍 故障排查

| 问题 | 排查方法 |
|------|---------|
| Label Studio 无法连接 | 检查 `curl http://localhost:8751/health` 是否返回 `{"status":"UP"}` |
| 预标注不生效 | 检查标签配置的 `name` 属性是否与 Backend 的 `from_name`/`to_name` 匹配 |
| API 调用失败 | 查看日志 `docker-compose logs -f`，检查 `MOONSHOT_API_KEY` 是否正确 |

## 📄 许可证

MIT
