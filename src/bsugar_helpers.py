import numpy as np


# Function to downcast float64 to float32
def downcast_float64_to_float32_and_objects_to_categories(df):
    # Downcast float64 columns to float32
    float64_cols = df.select_dtypes(include=['float64']).columns
    df[float64_cols] = df[float64_cols].astype(np.float32)
    
    # Convert object columns to category
    object_cols = df.select_dtypes(include=['object']).columns
    df[object_cols] = df[object_cols].astype('category')
    
    return df


def autoencode_columns_999(df, column_indices, encoding_dim=5, epochs=1, batch_size=32, autoencoder_id=0, train=True):
    print(f"Autoencoding columns from indices {column_indices[0]} to {column_indices[-1]} (feature set {autoencoder_id+1})")

    # Ensure that the number of columns (range) is exactly 216
    assert (column_indices[-1] - column_indices[0] + 1) == 216, \
        f"Expected 216 features (from column_indices {column_indices[0]} to {column_indices[-1]}), but got {(column_indices[-1] - column_indices[0] + 1)} columns. Please check your column_indices."

    # Extract the subset of columns based on indices
    data_subset = df.iloc[:, column_indices].values
    print(f"Original data shape for feature set {autoencoder_id+1}: {data_subset.shape}")
    
    # Impute missing values with the mean of the column
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(data_subset)

    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_imputed)
    
    if train:
        print("SIZE SIZE 999: ")
        # Autoencoder architecture
        input_dim = data_subset.shape[1]  # Dynamically setting input dimensions (should be 216)
        #print(input_dim)
        print(f"Autoencoder input dimension: {input_dim}")

        input_layer = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation='relu', 
                        activity_regularizer=regularizers.l1(1e-5))(input_layer)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)

        # Define the autoencoder model
        autoencoder = Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer='adam', loss='mse')

        # Train the autoencoder
        print(f"Training autoencoder {autoencoder_id}...")
        autoencoder.fit(data_scaled, data_scaled,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        verbose=1)

        # Save the trained autoencoder, scaler, and imputer
        autoencoder.save(os.path.join(AUTOENCODER_SAVE_DIR, f"autoencoder_{autoencoder_id}.keras"))
        joblib.dump(scaler, os.path.join(AUTOENCODER_SAVE_DIR, f"scaler_{autoencoder_id}.pkl"))
        joblib.dump(imputer, os.path.join(AUTOENCODER_SAVE_DIR, f"imputer_{autoencoder_id}.pkl"))
        print(f"Autoencoder {autoencoder_id}, scaler, and imputer saved.")

        # Create encoder model to extract the compressed representation
        encoder = Model(inputs=input_layer, outputs=encoded)
    else:
        # Load the trained autoencoder components for test data
        print(f"Loading autoencoder {autoencoder_id} for test data transformation...")
        autoencoder = load_model(os.path.join(AUTOENCODER_SAVE_DIR, f"autoencoder_{autoencoder_id}.keras"))
        scaler = joblib.load(os.path.join(AUTOENCODER_SAVE_DIR, f"scaler_{autoencoder_id}.pkl"))
        imputer = joblib.load(os.path.join(AUTOENCODER_SAVE_DIR, f"imputer_{autoencoder_id}.pkl"))
        
        # Impute and scale test data
        data_imputed = imputer.transform(data_subset)
        data_scaled = scaler.transform(data_imputed)

        # Use only the encoder part of the autoencoder
        encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(index=1).output)

    # Generate reduced data
    reduced_data = encoder.predict(data_scaled)
    print(f"Reduced data shape for feature set {autoencoder_id}: {reduced_data.shape}")

    # Create a DataFrame for the reduced features
    reduced_df = pd.DataFrame(reduced_data, 
                              columns=[f"autoencoded_{autoencoder_id}_{i}" for i in range(encoding_dim)])
    
    return reduced_df



def autoencode_columns(df, column_indices, encoding_dim=5, epochs=1, batch_size=32, autoencoder_id=0, train=True):
    print(f"Autoencoding columns")

 
    #print(f"Autoencoding columns {column_indices.start}-{column_indices.stop-1} (feature set {autoencoder_id+1})")

    # Extract the subset of columns
    data_subset = df.iloc[:, column_indices].values
    #print(f"Original data shape for feature set {autoencoder_id+1}: {data_subset.shape}")
    
    # Impute missing values with the mean of the column
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(data_subset)

    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_imputed)
    
    if train:
        print("SIZE SIZE")
        print(data_subset.shape[1])
        # Autoencoder architecture
        input_dim = data_subset.shape[1]
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation='relu', 
                        activity_regularizer=regularizers.l1(1e-5))(input_layer)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)

        # Define the autoencoder model
        autoencoder = Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer='adam', loss='mse')

        # Train the autoencoder
        print(f"Training autoencoder {autoencoder_id}...")
        autoencoder.fit(data_scaled, data_scaled,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        verbose=1)

        # Save the trained autoencoder, scaler, and imputer
        autoencoder.save(os.path.join(AUTOENCODER_SAVE_DIR, f"autoencoder_{autoencoder_id}.keras"))
        joblib.dump(scaler, os.path.join(AUTOENCODER_SAVE_DIR, f"scaler_{autoencoder_id}.pkl"))
        joblib.dump(imputer, os.path.join(AUTOENCODER_SAVE_DIR, f"imputer_{autoencoder_id}.pkl"))
        print(f"Autoencoder {autoencoder_id}, scaler, and imputer saved.")

        # Create encoder model to extract the compressed representation
        encoder = Model(inputs=input_layer, outputs=encoded)
    else:
        # Load the trained autoencoder components for test data
        print(f"Loading autoencoder {autoencoder_id} for test data transformation...")
        autoencoder = load_model(os.path.join(AUTOENCODER_SAVE_DIR, f"autoencoder_{autoencoder_id}.keras"))
        scaler = joblib.load(os.path.join(AUTOENCODER_SAVE_DIR, f"scaler_{autoencoder_id}.pkl"))
        imputer = joblib.load(os.path.join(AUTOENCODER_SAVE_DIR, f"imputer_{autoencoder_id}.pkl"))
        
        # Use only the encoder part of the autoencoder
        encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(index=1).output)

    # Generate reduced data
    reduced_data = encoder.predict(data_scaled)
    print(f"Reduced data shape for feature set {autoencoder_id}: {reduced_data.shape}")

    # Create a DataFrame for the reduced features
    reduced_df = pd.DataFrame(reduced_data, 
                              columns=[f"autoencoded_{autoencoder_id}_{i}" for i in range(encoding_dim)])
    
    return reduced_df
