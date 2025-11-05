from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

tokenizer = AutoTokenizer.from_pretrained("JetBrains-Research/cmg-race-without-history")
model = AutoModelForSeq2SeqLM.from_pretrained("JetBrains-Research/cmg-race-without-history")

# Example usage
commit = {
    "diff_id": 1180,
    "repo": "elastic/elasticsearch\n",
    "sha": "a397019139159dfc50d14e4826b857f25e4c5077\n",
    "time": "2014-11-13T18:25:46Z\n",
    "diff": "mmm a / README . md <nl> ppp b / README . md <nl> elasticsearch - license <nl> <nl> Internal Elasticsearch Licensing Plugin <nl> <nl> + # # Getting Started <nl> <nl> - # # Licensing REST APIs [ PUBLIC ] <nl> + see [ Getting Started ] ( https : / / github . com / elasticsearch / elasticsearch - license / blob / master / docs / getting - started . asciidoc ) <nl> + <nl> + # # Licensing REST APIs <nl> <nl> Licensing REST APIs enable users to : <nl> <nl> Licensing REST APIs enable users to : <nl> <nl> see [ Licensing REST APIs ] ( https : / / github . com / elasticsearch / elasticsearch - license / blob / master / docs / license . asciidoc ) <nl> <nl> - * * Note : * * see [ wiki ] ( https : / / github . com / elasticsearch / elasticsearch - license / wiki ) for documentation on * * Licensing Tools Usage & Reference * * , * * Licensing Consumer Plugin Interface * * , * * License Plugin Design * * and * * Licensing Test plan * * <nl> + # # Internal Documentation <nl> + see [ wiki ] ( https : / / github . com / elasticsearch / elasticsearch - license / wiki ) for documentation on <nl> + <nl> + - * * Licensing Tools Usage & Reference * * <nl> + - * * Licensing Consumer Plugin Interface * * <nl> + - * * License Plugin Design * * <nl> + - * * Licensing Test plan * * <nl>\n",
    "msg": "[ DOCS ] updated Readme\n",
}

inputs = tokenizer(commit["diff"], return_tensors="pt", max_length=512, truncation=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

outputs = model.generate(**inputs, max_length=30, num_beams=4)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated commit message:", result)
