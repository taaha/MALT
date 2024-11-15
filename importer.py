from datasets import load_dataset


hf_path = 'darthPanda/ur_en_questions'
dataset = load_dataset(hf_path)
    
dataset.save_to_disk("dataset/ur_en_questions")