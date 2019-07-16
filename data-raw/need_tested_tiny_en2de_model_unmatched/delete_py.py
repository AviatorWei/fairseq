paths = []
split = ['test']#],'valid','train']
lang = ['en']#,'de']
for i in split:
    for j in lang:
        paths.append(i+".en-de."+j)
for path in paths:
    with open(path, 'r') as fp:
        lines = fp.readlines()
    with open(path, 'w+') as wp:
        print("".join(list(filter(lambda x: len(x) > 0, lines))))