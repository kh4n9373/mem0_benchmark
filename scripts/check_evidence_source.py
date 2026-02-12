import json

def check_evidence_source():
    file_path = 'data/locomo/processed_data/longmemeval_processed_data.json'
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    ai_evidence_count = 0
    total_evidence_count = 0
    ai_evidence_category_counts = {}
    
    print(f"Loaded data with {len(data)} conversations.")

    for conv_idx, conv in enumerate(data):
        conv_id = conv.get('conv_id', f'idx_{conv_idx}')
        
        # Collect all AI messages in this conversation
        ai_messages = []
        user_messages = []
        
        for dialog in conv.get('dialogs', []):
            for msg in dialog.get('messages', []):
                role = msg.get('role', '').lower()
                content = msg.get('content', '')
                if role == 'assistant' or role == 'system' or role == 'model':
                    ai_messages.append(content)
                elif role == 'user':
                    user_messages.append(content)
        
        # Check evidences
        for qa in conv.get('qas', []):
            evidences = qa.get('evidences', [])
            for evidence in evidences:
                total_evidence_count += 1
                
                # Check if evidence matches (exactly or substring) any AI message
                found_in_ai = False
                for ai_msg in ai_messages:
                    if evidence in ai_msg:
                        found_in_ai = True
                        break
                
                if found_in_ai:
                    ai_evidence_count += 1
                    category = qa.get('category', 'unknown')
                    if category not in ai_evidence_category_counts:
                         ai_evidence_category_counts[category] = 0
                    ai_evidence_category_counts[category] += 1
                    
                    print(f"\n[MATCH] Evidence found in AI message for category '{category}'")
                    print(f"Conv ID: {conv_id}")
                    print(f"Question: {qa.get('question')}")
                    print(f"Evidence: {evidence}") 
                    # print(f"Matched AI snippet: ...")

    print(f"\nTotal evidences checked: {total_evidence_count}")
    print(f"Evidences appearing in AI messages: {ai_evidence_count}")
    print("Breakdown by category:")
    for cat, count in ai_evidence_category_counts.items():
        print(f"  {cat}: {count}")

if __name__ == "__main__":
    check_evidence_source()
