import csv
from matplotlib import pyplot as plt

with open('2020-04-13_malaria_b32.log') as csvf:
    reader = csv.DictReader(csvf)
    loss = []
    val_loss = []
    for row in reader:
        loss.append(float(row['loss']))
        val_loss.append(float(row['val_loss']))

plt.plot(loss)
plt.plot(val_loss)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('2020-04-13_loss_b32.png', format='png')