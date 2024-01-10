import os

from reprint import output


class MultiLineOutput:
    def __init__(self, num_lines, out):
        self.num_lines = num_lines
        self.lines = [{"label": "", "text": ""} for _ in range(num_lines)]
        self.out = out

    def refresh(self):
        self.out.parent.refresh()

    def append_line(self, line_idx, text):
        max_x = os.get_terminal_size().columns
        label = self.lines[line_idx]["label"]
        current_text = self.lines[line_idx]["text"]
        full_text = f"{label}{current_text}{text}"

        if len(full_text) >= max_x:
            # Clear the line and reset the text
            full_text = f"{label}{text}"
            self.lines[line_idx]["text"] = text
        else:
            self.lines[line_idx]["text"] += text

        line = self.lines[line_idx]
        self.out[line_idx] = f"{line['label']}{line['text']}"

    def set_label(self, line_idx, label):
        self.lines[line_idx]["label"] = label
        self.out[line_idx] = label

    def clear(self):
        for line_idx in range(self.num_lines):
            self.lines[line_idx] = {"label": "", "text": ""}
            self.out[line_idx] = ""


def main():
    import random
    import time

    print("Here is some normal terminal output.")

    with output(output_type="list", initial_len=32, interval=100) as out:
        mlo = MultiLineOutput(32, out)

        # initial labels for the lines
        lines = ["" for _ in range(mlo.num_lines)]
        for i in range(mlo.num_lines):
            mlo.set_label(i, f"Line {i}: ")

        # loop to continuously update the lines with new text
        while True:
            # generate random text for each line
            for i in range(mlo.num_lines):
                if i % 2 == 0:
                    mlo.append_line(
                        i, "".join(random.choices(["-", "+", "*", "/"], k=2))
                    )
                else:
                    mlo.append_line(
                        i, "".join(random.choices(["A", "B", "C", "D"], k=1))
                    )

            # wait a short time before updating again
            time.sleep(0.05)


if __name__ == "__main__":
    main()
