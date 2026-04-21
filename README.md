




## LLM
### aliyun LLM
```
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-plus",  # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': '你是谁？'}],
        stream=True,
        stream_options={"include_usage": True}
        )
    for chunk in completion:
        print(chunk.model_dump_json())
```
## 文生图
### 文生图Z-Image
```
curl --location 'https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation' \
--header 'Content-Type: application/json' \
--header "Authorization: Bearer $DASHSCOPE_API_KEY" \
--data '{
    "model": "z-image-turbo",
    "input": {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": "film grain, analog film texture, soft film lighting, Kodak Portra 400 style, cinematic grainy texture, photorealistic details, subtle noise, (film grain:1.2)。采用近景特写镜头拍摄的东亚年轻女性，呈现户外雪地场景。她体型纤瘦，呈站立姿势，身体微微向右侧倾斜，头部抬起看向画面上方，姿态自然放松。她的面部是典型东亚长相，肤色白皙，脸颊带有自然的红润感，五官清秀：眼睛是深棕色，眼型偏圆，眼神略带惊讶地望向上方，眼白部分可见；眉毛是深黑色，形状自然弯长；鼻子小巧挺直，嘴唇涂有红色口红，唇瓣微张，表情带着轻微的惊讶或好奇。她的头发是深黑色长直发，发丝被风吹得略显凌乱，部分垂在脸颊两侧，头顶佩戴一顶深灰色的头盔，头盔边缘露出少量发丝。服装是蓝白拼接的厚重外套，外套材质看起来是毛绒与布料结合，显得温暖厚实，适合雪地环境。背景是被白雪覆盖的户外场景，远处可见模糊的树木轮廓，天空是明亮的浅蓝色，带有少量白云，光线是强烈的自然日光，照亮人物面部与头发，形成清晰的光影，色调以蓝、白、黑为主，整体风格清新自然。画面顶部有黑色提示框，内有“Press esc to exit full screen”的白色文字。镜头的近景视角放大了人物的表情与细节，营造出户外雪地的真实氛围。"
                    }
                ]
            }
        ]
    },
    "parameters": {
        "prompt_extend": false,
        "size": "1120*1440"
    }
}'
```

## TTS
### 阿里云TTS
refer: https://bailian.console.aliyun.com/cn-beijing?spm=5176.12818093_47.overview_recent.1.5b8516d0G3WZAL&tab=api#/api/?type=model&url=2950054


## ASR
### 阿里云ASR
refer: https://bailian.console.aliyun.com/cn-beijing?spm=5176.12818093_47.overview_recent.1.5b8516d0G3WZAL&tab=api#/api/?type=model&url=2869148