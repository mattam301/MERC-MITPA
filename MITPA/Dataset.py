import numpy as np

def _labels_to_ints(labels):
    """
    Convert a list of labels to integer indices.
    Handles:
      - integer labels already: [0, 2, 1, ...]
      - one-hot lists/tuples/np arrays: [[0,1,0], [1,0,0], ...]
    Returns a list of ints.
    """
    ints = []
    for lab in labels:
        # If it's a scalar int/np.int_
        if isinstance(lab, (int, np.integer)):
            ints.append(int(lab))
        else:
            # try to treat as one-hot: find index of 1
            try:
                # convert to list for .index
                lab_list = list(lab)
                if 1 in lab_list:
                    ints.append(lab_list.index(1))
                else:
                    # fallback: argmax
                    ints.append(int(np.argmax(lab_list)))
            except Exception:
                # last resort: try to cast to int
                try:
                    ints.append(int(lab))
                except Exception:
                    raise ValueError("Unable to interpret label: %r" % (lab,))
    return ints

class Dataset:
    # ... your existing methods here ...

    def print_statistics(self, window_size=10, distinct_in_window=True, return_stats=False):
        """
        Print dataset statistics about emotion labels.

        Args:
          window_size (int): size of sub-dialogue window (default 10).
          distinct_in_window (bool): if True, compute number of DISTINCT labels per window.
                                     if False, compute number of LABELED utterances per window (useful
                                     if some utterances can be unlabeled / have a special 'no label').
          return_stats (bool): if True, return a dict with computed stats (in addition to printing).
        """
        max_per_conv = []
        mean_per_conv = []
        windows_counts = []  # holds counts per window across all conversations

        for s in self.samples:
            # convert labels to ints robustly
            labels = _labels_to_ints(s["labels"])
            if len(labels) == 0:
                # skip empty conversations
                continue

            arr = np.array(labels, dtype=int)

            max_per_conv.append(int(arr.max()))
            mean_per_conv.append(float(arr.mean()))

            # sliding windows of size window_size (non-overlapping or overlapping? we'll use sliding with stride 1)
            n = len(arr)
            if n <= 0:
                continue
            for start in range(0, max(1, n - window_size + 1)):
                window = arr[start : start + window_size]
                if distinct_in_window:
                    count = int(len(set(window.tolist())))
                else:
                    # count non-None / non -1 labels (if your unlabeled marker is -1, change check accordingly)
                    # here we assume every item is a label; adjust if you have sentinel for missing labels
                    count = int(np.sum([1 for x in window if x is not None]))
                windows_counts.append(count)

            # handle last short tail windows if conversation shorter than window_size or to include tail:
            # (Optional) include the final shorter window as well:
            if n < window_size:
                if distinct_in_window:
                    windows_counts.append(int(len(set(arr.tolist()))))
                else:
                    windows_counts.append(int(len(arr)))

        # Now compute summary stats and print
        def _summ(stats):
            if len(stats) == 0:
                return {"mean": None, "min": None, "max": None}
            return {"mean": float(np.mean(stats)), "min": int(np.min(stats)), "max": int(np.max(stats))}

        conv_max_stats = _summ(max_per_conv)
        conv_mean_stats = _summ(mean_per_conv)
        window_stats = _summ(windows_counts)

        print("=== Dataset label statistics ===")
        print(f"Number of conversations: {len(self.samples)}")
        print()
        print("Per-conversation:")
        print(f"  - max label per conversation: mean={conv_max_stats['mean']:.4f}  min={conv_max_stats['min']}  max={conv_max_stats['max']}")
        print(f"  - mean label per conversation: mean={conv_mean_stats['mean']:.4f}  min={conv_mean_stats['min']}  max={conv_mean_stats['max']}")
        print()
        mode = "distinct labels per window" if distinct_in_window else "labeled utterances per window"
        print(f"Sliding-window (size={window_size}) statistics ({mode}):")
        if window_stats["mean"] is None:
            print("  No windows found (dataset empty or no labels).")
        else:
            print(f"  - mean = {window_stats['mean']:.4f}")
            print(f"  - min  = {window_stats['min']}")
            print(f"  - max  = {window_stats['max']}")
        print("================================")

        stats_out = {
            "num_conversations": len(self.samples),
            "conv_max_per_conv": max_per_conv,
            "conv_mean_per_conv": mean_per_conv,
            "conv_max_summary": conv_max_stats,
            "conv_mean_summary": conv_mean_stats,
            "windows_counts": windows_counts,
            "windows_summary": window_stats,
            "window_size": window_size,
            "distinct_in_window": distinct_in_window,
        }

        if return_stats:
            return stats_out