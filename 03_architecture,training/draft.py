# LLAMA 13B
B = 8
N = 2048
D = 5120
h = 40
d = D // h

train_flops = B * N * D * D + B * N * N * D
infer_flops = B * N * D * D + B * N * N * D
train_bytes = B * N * D + D * D + B * h * N * N
infer_bytes = B * N * N * D + N * D * D

train_AI = train_flops / train_bytes
infer_AI = infer_flops / infer_bytes
print(f"Train AI: {train_AI:.2f} FLOPs/Byte")  # 414.04
print(f"Infer AI: {infer_AI:.2f} FLOPs/Byte")  # 2.67

# 引入MQA
infer_bytes = B * N * N * d + N * D * D
infer_AI = infer_flops / infer_bytes
print(f"Infer AI with MQA: {infer_AI:.2f} FLOPs/Byte")  # 10.37

# Large batch size when inferencing
B = 128
infer_flops = B * N * D * D + B * N * N * D
infer_bytes = B * N * N * d + N * D * D
infer_AI = infer_flops / infer_bytes
print(f"Infer AI with MQA: {infer_AI:.2f} FLOPs/Byte")  # 78.60