import csv
import matplotlib.pyplot as plt

def plot_line_chart(csv_file,fig_name):
    x = []
    y = []
    plt.figure()

    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            x.append(row[1])
            y.append(row[2])
    x.pop(0)
    y.pop(0)

    xx = [eval(i) for i in x]
    yy = [eval(i) for i in y]
    plt.plot(xx, yy)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward Chart')
    # plt.show()
    plt.savefig(fig_name)

def plot_dis_chart(csv_file,fig_name):
    x = []
    y = []
    plt.figure()

    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            x.append(row[1])
            y.append(row[2])
    x.pop(0)
    y.pop(0)

    xx = [eval(i) for i in x]
    yy = [eval(i) for i in y]
    plt.plot(xx, yy)
    plt.xlabel('Episode')
    plt.ylabel('Distance')
    plt.title('Reward Chart')
    # plt.show()
    plt.savefig(fig_name)

def plot_loss_chart(csv_file,fig_name,loss_name = "Actor loss"):
    x = []
    y = []
    plt.figure()

    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            x.append(row[1])
            y.append(row[2])
    x.pop(0)
    y.pop(0)

    xx = [eval(i) for i in x]
    yy = [eval(i) for i in y]
    plt.plot(xx, yy, label = loss_name)

    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title(loss_name+'Chart')
    plt.legend()
    # plt.show()
    plt.savefig(fig_name)

