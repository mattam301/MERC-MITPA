import pickle
import os

def convert_new_to_old_format(new_data_path, old_data_path):
    with open(new_data_path, 'rb') as f:
        new_data = pickle.load(f)
    print("Loaded new format data.")
    print(len(new_data))
    for i in range(len(new_data)):
        print(f"Sample {i}: {type(new_data[i])}.")
        if isinstance(new_data[i], dict):
            # The line `print(f"Keys: {list(new_data[i].keys())}")` is printing out the keys of a
            # dictionary stored in the `new_data` list at index `i`.
            # print(f"Keys: {list(new_data[i].keys())}")
            print(f"Sample values: {new_data[i]["Ses02F_script01_3"]}")
        elif isinstance(new_data[i], list):
            print(f"Length of list: {len(new_data[i])}, sample: {new_data[i][:2]}")
    (
        videoIDs,
        videoSpeakers,
        videoLabels,
        videoText,
        videoText1,
        videoText2,
        videoText3,
        videoAudio,
        videoVisual,
        videoSentence,
        trainVid,
        testVid,
    ) = new_data

    old_data = {'train': [], 'test': []}

    for split_name, splitVidList in [('train', trainVid), ('test', testVid)]:
        for vid in splitVidList:
            sample = {
                'vid': vid,
                'speakers': videoSpeakers[vid],
                'labels': videoLabels[vid],
                'text': videoText[vid],
                'audio': videoAudio[vid],
                'visual': videoVisual[vid],
                'sentence': videoSentence.get(vid, "")
            }
            old_data[split_name].append(sample)

    os.makedirs(os.path.dirname(old_data_path), exist_ok=True)
    with open(old_data_path, 'wb') as f:
        pickle.dump(old_data, f)

    print(f"Converted data saved to {old_data_path}")
    
convert_new_to_old_format("/Users/tuanmai/Downloads/data/iemocap_multi_features.pkl", "data/iemocap_gs/data_iemocap_gs.pkl")