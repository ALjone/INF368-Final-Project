from GPT2Tuner import GPT2Tuner

class Lambada:
    def __init__(self, data_path, G_epochs = 20, h_epochs = 20, sentences_per_label = 100, save_path = "samples.txt") -> None:
        self.G = GPT2Tuner(data_path)
        self.h = None
        for _ in range(G_epochs):
            self.G.train_epoch()
        for i in range(int((sentences_per_label*10)/50)):
            self.G.save_sentences(50, save_path)

        with open(save_path, "r") as file:
            sentences = file.readlines()
        labels = [s[0:3] for s in sentences]
        sentences = [s[0:10] for s in sentences]

        predictions = self.h.predict(sentences)
        