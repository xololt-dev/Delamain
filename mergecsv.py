import csv
import sys


def merge_csv_by_columns(file1, file2, output_file, label=None):
    """
    Merges two CSV files side-by-side.
    Keeps the labels from file1, adds values from file1,
    and appends only the value columns from file2.

    If `label` is provided, finds the row matching that label in both files
    and skips any column from file2 whose value at that row already exists
    in file1's reference row.
    """
    try:
        with open(file1, "r", newline="", encoding="utf-8") as f1, open(
            file2, "r", newline="", encoding="utf-8"
        ) as f2:

            reader1 = csv.reader(f1)
            reader2 = csv.reader(f2)
            rows1 = list(reader1)
            rows2 = list(reader2)

        skip_indices = set()

        if label is not None:
            label_str = str(label)
            ref_row1 = next((r for r in rows1 if r and str(r[0]) == label_str), None)
            ref_row2 = next((r for r in rows2 if r and str(r[0]) == label_str), None)

            if ref_row1 is None:
                print(f"Error: label '{label}' not found in file1")
                return
            if ref_row2 is None:
                print(f"Error: label '{label}' not found in file2")
                return

            csv1_vals = set(ref_row1[1:])
            for i in range(1, len(ref_row2)):
                if ref_row2[i] in csv1_vals:
                    skip_indices.add(i)

        merged_rows = []

        for row1, row2 in zip(rows1, rows2):
            combined_row = row1 + [
                row2[i] for i in range(1, len(row2)) if i not in skip_indices
            ]
            merged_rows.append(combined_row)

        with open(output_file, "w", newline="", encoding="utf-8") as out_f:
            writer = csv.writer(out_f)
            writer.writerows(merged_rows)

        print(
            f"Successfully merged! '{output_file}' created with {len(merged_rows)} rows."
        )
        if label is not None and skip_indices:
            print(
                f"  Skipped {len(skip_indices)} duplicate column(s) from file2 "
                f"based on label '{label}'."
            )

    except FileNotFoundError as e:
        print(f"Error: {e}")


def merge_csv_by_columns_2(file1, file2, output_file, label=None):
    """
    Merges two CSV files side-by-side.
    Keeps the labels from file1, adds values from file1,
    and appends only the value columns from file2.

    If `label` is provided, finds the row matching that label in both files
    and skips any column from file2 whose value at that row already exists
    in file1's reference row.
    """
    try:
        with open(file1, "r", newline="", encoding="utf-8") as f1, open(
            file2, "r", newline="", encoding="utf-8"
        ) as f2:

            reader1 = csv.reader(f1)
            reader2 = csv.reader(f2)
            rows1 = list(reader1)
            rows2 = list(reader2)

        skip_indices = set()
        max_csv1_index = 1

        if label is not None:
            label_str = str(label)
            ref_row1 = next((r for r in rows1 if r and str(r[0]) == label_str), None)
            ref_row2 = next((r for r in rows2 if r and str(r[0]) == label_str), None)

            if ref_row1 is None:
                print(f"Error: label '{label}' not found in file1")
                return
            if ref_row2 is None:
                print(f"Error: label '{label}' not found in file2")
                return

            csv2_start_val = ref_row2[1]
            for row in ref_row1[1:]:
                if (
                    float(row) > float(csv2_start_val) * 0.9995
                    and float(row) < float(csv2_start_val) * 1.0005
                ):
                    print(f"connection at index: {max_csv1_index}")
                    break
                max_csv1_index += 1

        merged_rows = []

        for row1, row2 in zip(rows1, rows2):
            combined_row = row1[:max_csv1_index] + row2[1:]
            merged_rows.append(combined_row)

        with open(output_file, "w", newline="", encoding="utf-8") as out_f:
            writer = csv.writer(out_f)
            writer.writerows(merged_rows)

        print(
            f"Successfully merged! '{output_file}' created with {len(merged_rows)} rows."
        )
        if label is not None and skip_indices:
            print(
                f"  Skipped {len(skip_indices)} duplicate column(s) from file2 "
                f"based on label '{label}'."
            )

    except FileNotFoundError as e:
        print(f"Error: {e}")


# --- Example Usage ---
if __name__ == "__main__":
    file_a = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "./DQN base Del2.6/Delamain_2_6_log_test.csv"
    )
    file_b = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "./DQN base Del2.6/Delamain_2_6_log_test_1.csv"
    )
    output = (
        sys.argv[3]
        if len(sys.argv) > 3
        else "./DQN base Del2.6/Delamain_2_6_log_test_dest.csv"
    )
    label = sys.argv[4] if len(sys.argv) > 4 else None
    print(label)
    merge_csv_by_columns_2(file_a, file_b, output, label)
