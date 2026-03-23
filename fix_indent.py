def fix(f):
    with open(f, 'r') as file:
        lines = file.readlines()
    for i in range(len(lines)):
        if "logits_m = (logits_m - logits_m.mean" in lines[i]:
            spaces = len(lines[i-2]) - len(lines[i-2].lstrip())
            lines[i] = " " * spaces + lines[i].lstrip()
        if "labels_m = labels[" in lines[i] and i > 0 and "logits_m" in lines[i-1]:
            spaces = len(lines[i-3]) - len(lines[i-3].lstrip())
            lines[i] = " " * spaces + lines[i].lstrip()

    with open(f, 'w') as file:
        file.writelines(lines)

fix('run_bert.py')
fix('run_mlm.py')
