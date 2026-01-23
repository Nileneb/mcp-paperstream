from transformers import AutoTokenizer, AutoModel

# Wird automatisch nach ~/.cache/huggingface/ gecached
tokenizer = AutoTokenizer.from_pretrained("nlpie/distil-biobert")
model = AutoModel.from_pretrained("nlpie/distil-biobert")

# UND in eigenen Ordner speichern:
tokenizer.save_pretrained("../models/biobert/distil-biobert")
model.save_pretrained("../models/biobert/distil-biobert")
