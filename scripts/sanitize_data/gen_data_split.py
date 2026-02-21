import os, random

# get amounts for each split
cropped_path = os.path.join(".", "image_cropped")
total = sum(1 for name in os.listdir(cropped_path) if os.path.isfile(os.path.join(cropped_path, name)))
print(f"Total: {total}")
split = ["train", "validation", "test"]
amount = [int(total / 10 * 7), int(total / 10 * 1), int(total / 10 * 2)]
print(f"Split amounts: {amount}, sum: {sum(amount)}")
assert(sum(amount) <= total)

# randomize split indices
choices = []
for i in range(len(amount)):
    choices += [i] * amount[i]
random.shuffle(choices)

# write to files
files = [""] * 3
for i in range(len(choices)):
    curr_choice = choices[i]
    files[curr_choice] += f"{i:04d}\n"
for i in range(len(split)):
    dir_path = os.path.join(".", "data_split_dirs", split[i] + ".txt")
    if not os.path.exists(os.path.dirname(dir_path)):
        os.makedirs(os.path.dirname(dir_path))
    with open(dir_path, "w") as f:
        f.write(files[i])

print("Complete.")
