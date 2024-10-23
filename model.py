import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

# Определение класса для датасета
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Определение модели
class FullyConnectedNN(nn.Module):
    def __init__(self, input_size):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))  # Выходной слой для бинарной классификации
        return x

# Функция для обучения модели
def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Функция для оценки модели
def evaluate_model(model, val_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            y_true.extend(labels.numpy())
            y_pred.extend(outputs.squeeze().numpy())
    
    auc = roc_auc_score(y_true, y_pred)
    print(f'ROC-AUC: {auc:.4f}')
    return auc

# Функция для предсказания на новых данных
def predict(model, X_new):
    model.eval()
    with torch.no_grad():
        X_new_tensor = torch.tensor(X_new.values, dtype=torch.float32)
        outputs = model(X_new_tensor)
        probabilities = outputs.squeeze().numpy()

    # Преобразование всех значений в float и форматирование до 6 знаков после запятой
    probabilities = np.array([float("{:.6f}".format(prob)) for prob in probabilities])
    return probabilities

# Функция для сохранения модели
def save_model(model, file_path='trained_model.pth'):
    torch.save(model.state_dict(), file_path)
    print(f'Модель сохранена в файл {file_path}')

# Функция для загрузки модели
def load_model(model, file_path='trained_model.pth'):
    model.load_state_dict(torch.load(file_path, weights_only=True))
    model.eval()
    print(f'Модель загружена')
    return model

# Блок для обучения модели
if __name__ == "__main__":
    from data_preprocessing import FeatureExtractor
    import pandas as pd

    # Загрузка и подготовка данных
    df = pd.read_parquet('test.parquet')
    
    extractor = FeatureExtractor()
    df_prepared = extractor.prepare_data(df)

    # Извлечение признаков
    features = extractor.extract_and_process_features(df_prepared)

    # Разделение на признаки и метки (предполагается, что 'label' - целевая переменная)
    X = features.drop(columns=['label'])  # Замените 'label' на вашу целевую переменную
    y = features['label']

    # Создание DataLoader
    train_dataset = CustomDataset(X, y)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Инициализация модели, функции потерь и оптимизатора
    input_size = X.shape[1]
    model = FullyConnectedNN(input_size)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Обучение модели
    train_model(model, train_loader, criterion, optimizer, num_epochs=75)

    # Сохранение модели
    save_model(model)

