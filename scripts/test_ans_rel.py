from sentence_transformers import (
    SentenceTransformer,
    util
)

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

a = "Avery Lancaster is the co-founder and CEO of Insurellm."

b = "Avery Lancaster is the co-founder and CEO of Insurellm."

emb1 = model.encode(
    a,
    convert_to_tensor=True
)

emb2 = model.encode(
    b,
    convert_to_tensor=True
)

print(
    util.cos_sim(
        emb1,
        emb2
    ).item()
)