import math
import random
import torch
import numpy as np
from collections import Counter

def revert_one_hot(one_hot_list):
    normal_numbers = []
    for one_hot in one_hot_list:
        normal_numbers.append(one_hot.index(1))
    return normal_numbers

class Dataset:
    def __init__(self, samples, args) -> None:
        self.samples = samples
        self.batch_size = args.batch_size
        self.num_batches = math.ceil(len(self.samples) / args.batch_size)
        self.dataset = args.dataset
        self.speaker_to_idx = {"M": 0, "F": 1}
        self.embedding_dim = args.dataset_embedding_dims[args.dataset]

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        batch = self.raw_batch(index)
        return self.padding(batch)

    def raw_batch(self, index):
        assert index < self.num_batches, "batch_idx %d > %d" % (index, self.num_batches)
        batch = self.samples[index * self.batch_size : (index + 1) * self.batch_size]
        return batch

    def padding(self, samples):
        # ... (your existing padding method) ...
        batch_size = len(samples)
        text_len_tensor = torch.tensor([len(s["text"]) for s in samples]).long()
        mx = torch.max(text_len_tensor).item()
        
        text_tensor = torch.zeros((batch_size, mx, self.embedding_dim['t']))
        audio_tensor = torch.zeros((batch_size, mx, self.embedding_dim['a']))
        visual_tensor = torch.zeros((batch_size, mx, self.embedding_dim['v']))
        speaker_tensor = torch.zeros((batch_size, mx)).long()
        labels = []
        utterances = []
        
        for i, s in enumerate(samples):
            cur_len = len(s["text"])
            utterances.append(s["sentence"])
            
            # Stack modality features
            tmp_t = torch.stack([torch.tensor(t) for t in s["text"]])
            tmp_a = torch.stack([torch.tensor(a) for a in s["audio"]])
            tmp_v = torch.stack([torch.tensor(v) for v in s["visual"]])
            
            text_tensor[i, :cur_len, :] = tmp_t
            audio_tensor[i, :cur_len, :] = tmp_a
            visual_tensor[i, :cur_len, :] = tmp_v

            # Handle speakers
            if self.dataset == "iemocap_roberta" or self.dataset == "mosei":
                speaker_tensor[i, :cur_len] = torch.tensor(s["speakers"])
            elif self.dataset == "meld":
                speaker_tensor[i, :cur_len] = torch.Tensor(revert_one_hot(s['speakers']))
            else:
                speaker_tensor[i, :cur_len] = torch.tensor([self.speaker_to_idx[c] for c in s["speakers"]])

            labels.extend(s["labels"])

        label_tensor = torch.tensor(labels).long()
        
        data = {
            "text_len_tensor": text_len_tensor,
            "text_tensor": text_tensor,
            "audio_tensor": audio_tensor,
            "visual_tensor": visual_tensor,
            "speaker_tensor": speaker_tensor,
            "label_tensor": label_tensor,
            "utterance_texts": utterances,
        }
        return data

    def shuffle(self):
        random.shuffle(self.samples)

    def print_statistics(self, window_size=10):
        """
        Print comprehensive statistics about emotion labels in the dataset.
        
        Statistics include:
        - Global label distribution
        - Per-conversation label extremes and averages
        - Sub-dialogue analysis (emotion diversity in windows of N utterances)
        
        Args:
            window_size: Size of sub-dialogue windows (default: 10)
        """
        print("\n" + "=" * 60)
        print(f"Dataset Statistics ({self.dataset})")
        print("=" * 60)
        
        # Basic dataset info
        total_conversations = len(self.samples)
        total_utterances = sum(len(s.get("labels", [])) for s in self.samples)
        print(f"Total conversations: {total_conversations}")
        print(f"Total utterances: {total_utterances}")
        if total_conversations > 0:
            print(f"Average utterances per conversation: {total_utterances / total_conversations:.2f}")
        
        # Collect all labels and per-conversation stats
        all_labels = []
        conv_stats = []
        
        for s in self.samples:
            labels = s.get("labels", [])
            if not labels:
                continue
                
            all_labels.extend(labels)
            
            # Per-conversation statistics
            conv_stats.append({
                'max_label': max(labels),  # Highest label value in conversation
                'avg_label': sum(labels) / len(labels),  # Average label value
                'unique_labels': len(set(labels)),  # Number of distinct emotions
                'length': len(labels)
            })
        
        if not all_labels:
            print("No labels found in dataset!")
            return
        
        # 1. Global label statistics
        print("\n--- Global Label Statistics ---")
        print(f"Global max label value: {max(all_labels)}")
        print(f"Global min label value: {min(all_labels)}")
        print(f"Global average label value: {sum(all_labels) / len(all_labels):.4f}")
        
        # Label distribution
        overall_counts = Counter(all_labels)
        print(f"\nLabel distribution:")
        for label in sorted(overall_counts.keys()):
            count = overall_counts[label]
            percentage = (count / len(all_labels)) * 100
            print(f"  Label {label}: {count} utterances ({percentage:.2f}%)")
        
        # 2. Per-conversation statistics
        print("\n--- Per-Conversation Statistics ---")
        print(f"Average conversation length: {sum(s['length'] for s in conv_stats) / len(conv_stats):.2f}")
        print(f"Average of conversation max labels: {sum(s['max_label'] for s in conv_stats) / len(conv_stats):.4f}")
        print(f"Average of conversation avg labels: {sum(s['avg_label'] for s in conv_stats) / len(conv_stats):.4f}")
        print(f"Average unique labels per conversation: {sum(s['unique_labels'] for s in conv_stats) / len(conv_stats):.2f}")
        
        # 3. Sub-dialogue analysis (windows of 10 utterances)
        print(f"\n--- Sub-dialogue Statistics (window size = {window_size}) ---")
        sub_dialogue_stats = []
        
        for conv_idx, s in enumerate(self.samples):
            labels = s.get("labels", [])
            if len(labels) < 2:
                continue
            
            # Create non-overlapping windows
            for start in range(0, len(labels), window_size):
                end = min(start + window_size, len(labels))
                window_labels = labels[start:end]
                
                # Skip windows smaller than half the window size
                if len(window_labels) < window_size // 2:
                    continue
                
                unique_count = len(set(window_labels))
                label_counts = Counter(window_labels)
                
                sub_dialogue_stats.append({
                    'conversation_id': conv_idx,
                    'size': len(window_labels),
                    'unique_labels': unique_count,
                    'most_frequent_count': max(label_counts.values()),
                    'least_frequent_count': min(label_counts.values()),
                })
        
        if sub_dialogue_stats:
            print(f"Total sub-dialogues analyzed: {len(sub_dialogue_stats)}")
            
            # Unique labels per sub-dialogue (emotion diversity)
            unique_counts = [stat['unique_labels'] for stat in sub_dialogue_stats]
            print(f"\nUnique emotion labels per sub-dialogue:")
            print(f"  Average: {sum(unique_counts) / len(unique_counts):.2f}")
            print(f"  Max: {max(unique_counts)}")
            print(f"  Min: {min(unique_counts)}")
            
            # Most frequent label count per sub-dialogue (dominant emotion)
            most_frequent_counts = [stat['most_frequent_count'] for stat in sub_dialogue_stats]
            print(f"\nDominant emotion frequency per sub-dialogue:")
            print(f"  Average: {sum(most_frequent_counts) / len(most_frequent_counts):.2f}")
            print(f"  Max: {max(most_frequent_counts)}")
            print(f"  Min: {min(most_frequent_counts)}")
            
        else:
            print("No sub-dialogues found (conversations too short)")
        
        print("=" * 60 + "\n")