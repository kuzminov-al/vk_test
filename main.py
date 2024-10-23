if __name__ == "__main__":
    import pandas as pd
    from data_preprocessing import FeatureExtractor
    from model import FullyConnectedNN, load_model, predict

    # Загрузка и подготовка данных
    df = pd.read_parquet('test.parquet')
    
    extractor = FeatureExtractor()
    df_prepared = extractor.prepare_data(df)

    # Извлечение и обработка признаков
    features = extractor.extract_and_process_features(df_prepared)
    
    # Инициализация модели
    input_size = features.shape[1]  
    model = FullyConnectedNN(input_size)

    # Загрузка обученной модели
    model = load_model(model, file_path='trained_model.pth')

    # Предсказание вероятностей для класса 1
    probabilities = predict(model, features)

    results = pd.DataFrame({
        'id': features.index, 
        'score': probabilities  
    })

    # Сохранение результатов в CSV файл
    results.to_csv('submission.csv', index=False)

    print("Предсказания сохранены в 'submission.csv'")
