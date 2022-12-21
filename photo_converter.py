from PIL import Image
alphabet =['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
for abc in alphabet:
    for i in range(20):
        a = i + 1
        img = Image.open(f'greyscale/{abc}/{abc}{a}.png').convert('L')
        img.save(f'greyscale/{abc}/{abc}{a}.png')

