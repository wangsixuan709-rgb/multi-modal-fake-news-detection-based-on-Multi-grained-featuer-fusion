with open('/root/autodl-tmp/MMFN_yyt/weibo_dataset.py', 'r') as f:
    content = f.read()

old_str = """token = BertTokenizer.from_pretrained('bert-base-chinese', local_files_only=True)"""
new_str = """token = BertTokenizer.from_pretrained('bert-base-chinese', local_files_only=True)
entity_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', local_files_only=True)
entity_enricher = WikiEntityEnricher()"""

if old_str in content:
    with open('/root/autodl-tmp/MMFN_yyt/weibo_dataset.py', 'w') as f:
        f.write(content.replace(old_str, new_str))
    print("Replace done 1")

old_str2 = """    textclip = clip.tokenize(textclip, truncate=True)
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    image = torch.stack(image).squeeze()
    imageclip = torch.stack(imageclip)
    labels = torch.LongTensor(labels)
    return input_ids, attention_mask, token_type_ids, image, imageclip, textclip, labels"""

new_str2 = """    textclip = clip.tokenize(textclip, truncate=True)
    
    use_entity_enrich = os.environ.get("USE_ENTITY_ENRICH", "0") == "1"
    if use_entity_enrich:
        enrich_texts = [entity_enricher.build_background_text(sent) for sent in sents]
    else:
        enrich_texts = ["" for _ in sents]

    entity_data = entity_tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=enrich_texts,
        truncation=True,
        padding='max_length',
        max_length=64,
        return_tensors='pt',
        return_length=True
    )

    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    image = torch.stack(image).squeeze()
    imageclip = torch.stack(imageclip)
    labels = torch.LongTensor(labels)
    return input_ids, attention_mask, token_type_ids, image, imageclip, textclip, labels, entity_data['input_ids'], entity_data['attention_mask'], entity_data['token_type_ids']"""

if old_str2 in content:
    content = content.replace(old_str, new_str)
    with open('/root/autodl-tmp/MMFN_yyt/weibo_dataset.py', 'w') as f:
        f.write(content.replace(old_str2, new_str2))
    print("Replace done 2")

