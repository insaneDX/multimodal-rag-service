from src.retriever import hybrid_retrieve, build_prompt
res = hybrid_retrieve("what are symptoms of pneumonia?", top_k_text=5, top_k_image=3)
prompt = build_prompt("what are symptoms of pneumonia?", res, top_k=6)
print(prompt)
