import argparse

parser = argparse.ArgumentParser()
opt = parser.parse_args()

dataset = data_loader.load_data()

print("[INFO] training images : {}".format(len(data_loader)))

model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0

