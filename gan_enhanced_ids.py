"""
GAN-Enhanced Network Intrusion Detection System
Uses Generative Adversarial Networks for data augmentation and advanced ML
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import json
import warnings
warnings.filterwarnings('ignore')

class NetworkGAN:
    """GAN for generating synthetic network traffic data"""
    
    def __init__(self, input_dim, noise_dim=100):
        self.input_dim = input_dim
        self.noise_dim = noise_dim
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.gan = self._build_gan()
        
    def _build_generator(self):
        """Build the generator network"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_dim=self.noise_dim),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(self.input_dim, activation='tanh')  # Output layer
        ])
        
        return model
    
    def _build_discriminator(self):
        """Build the discriminator network"""
        model = keras.Sequential([
            layers.Dense(512, activation='relu', input_dim=self.input_dim),
            layers.Dropout(0.4),
            
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.4),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.4),
            
            layers.Dense(1, activation='sigmoid')  # Binary output
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(0.0002, 0.5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_gan(self):
        """Build the combined GAN model"""
        # Make discriminator non-trainable when training generator
        self.discriminator.trainable = False
        
        model = keras.Sequential([
            self.generator,
            self.discriminator
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(0.0002, 0.5),
            loss='binary_crossentropy'
        )
        
        return model
    
    def train(self, X_real, epochs=1000, batch_size=32):
        """Train the GAN"""
        print(f"Training GAN for {epochs} epochs...")
        
        # Normalize data to [-1, 1] range for tanh activation
        X_real = 2 * (X_real - X_real.min()) / (X_real.max() - X_real.min()) - 1
        
        # Ground truth labels
        valid_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        
        d_losses = []
        g_losses = []
        
        for epoch in range(epochs):
            # Train Discriminator
            # Select random real samples
            idx = np.random.randint(0, X_real.shape[0], batch_size)
            real_samples = X_real[idx]
            
            # Generate fake samples
            noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
            fake_samples = self.generator.predict(noise, verbose=0)
            
            # Train discriminator on real and fake data
            d_loss_real = self.discriminator.train_on_batch(real_samples, valid_labels)
            d_loss_fake = self.discriminator.train_on_batch(fake_samples, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train Generator
            noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
            g_loss = self.gan.train_on_batch(noise, valid_labels)
            
            d_losses.append(d_loss[0])
            g_losses.append(g_loss)
            
            # Print progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: D Loss: {d_loss[0]:.4f}, G Loss: {g_loss:.4f}")
        
        print("GAN training completed!")
        return d_losses, g_losses
    
    def generate_samples(self, num_samples):
        """Generate synthetic samples"""
        noise = np.random.normal(0, 1, (num_samples, self.noise_dim))
        synthetic_data = self.generator.predict(noise, verbose=0)
        
        # Denormalize from [-1, 1] to [0, 1]
        synthetic_data = (synthetic_data + 1) / 2
        
        return synthetic_data

class GANEnhancedIDS:
    """GAN-Enhanced Intrusion Detection System"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.gan_models = {}
        self.classifier = None
        
    def load_data(self):
        """Load the network traffic data"""
        try:
            df = pd.read_csv('enriched_network_traffic.csv')
            print(f"Loaded {len(df)} network traffic records")
            return df
        except FileNotFoundError:
            print("Generating synthetic data for GAN training...")
            return self._generate_base_data()
    
    def _generate_base_data(self):
        """Generate base synthetic data"""
        np.random.seed(42)
        
        data = []
        attack_types = ['Normal', 'DDoS', 'Port Scan', 'Brute Force', 'Web Attack']
        samples_per_type = [8000, 1500, 800, 600, 600]  # Imbalanced dataset
        
        for attack_type, num_samples in zip(attack_types, samples_per_type):
            for _ in range(num_samples):
                if attack_type == 'Normal':
                    record = {
                        'duration': np.random.exponential(30),
                        'packet_count': np.random.poisson(50),
                        'bytes_transferred': np.random.lognormal(10, 1),
                        'packets_per_second': np.random.exponential(10),
                        'bytes_per_second': np.random.lognormal(12, 1),
                        'packet_size_avg': np.random.normal(800, 200),
                        'src_ip_reputation': np.random.uniform(0.8, 1.0),
                        'anomaly_score': np.random.uniform(0.0, 0.2),
                        'src_ip_malicious': 0,
                        'dst_ip_malicious': 0,
                        'dst_port': np.random.choice([80, 443, 53, 25]),
                        'label': 'Normal'
                    }
                elif attack_type == 'DDoS':
                    record = {
                        'duration': np.random.exponential(1),
                        'packet_count': np.random.poisson(1000),
                        'bytes_transferred': np.random.lognormal(8, 1),
                        'packets_per_second': np.random.exponential(1000),
                        'bytes_per_second': np.random.lognormal(10, 1),
                        'packet_size_avg': np.random.normal(100, 50),
                        'src_ip_reputation': np.random.uniform(0.1, 0.3),
                        'anomaly_score': np.random.uniform(0.8, 1.0),
                        'src_ip_malicious': 1,
                        'dst_ip_malicious': 0,
                        'dst_port': np.random.choice([80, 443, 53]),
                        'label': 'DDoS'
                    }
                else:  # Other attacks
                    record = {
                        'duration': np.random.exponential(5),
                        'packet_count': np.random.poisson(100),
                        'bytes_transferred': np.random.lognormal(9, 1),
                        'packets_per_second': np.random.exponential(50),
                        'bytes_per_second': np.random.lognormal(11, 1),
                        'packet_size_avg': np.random.normal(400, 100),
                        'src_ip_reputation': np.random.uniform(0.2, 0.5),
                        'anomaly_score': np.random.uniform(0.6, 0.9),
                        'src_ip_malicious': np.random.choice([0, 1]),
                        'dst_ip_malicious': 0,
                        'dst_port': np.random.randint(1, 65536),
                        'label': attack_type
                    }
                
                data.append(record)
        
        return pd.DataFrame(data)
    
    def preprocess_data(self, df):
        """Preprocess data for ML"""
        # Drop non-numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'label' in df.columns:
            X = df[numeric_cols]
            y = df['label']
        else:
            X = df[numeric_cols]
            y = None
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        if y is not None:
            y_encoded = self.label_encoder.fit_transform(y)
            return X_scaled, y_encoded, X.columns
        else:
            return X_scaled, None, X.columns
    
    def augment_minority_classes(self, X, y, target_samples=2000):
        """Use GANs to augment minority classes"""
        print("Augmenting minority classes with GANs...")
        
        unique_classes, class_counts = np.unique(y, return_counts=True)
        augmented_X = [X]
        augmented_y = [y]
        
        for class_idx, count in zip(unique_classes, class_counts):
            class_name = self.label_encoder.classes_[class_idx]
            
            if count < target_samples and class_name != 'Normal':  # Don't augment majority class
                print(f"Augmenting {class_name} from {count} to {target_samples} samples")
                
                # Get class data
                class_mask = y == class_idx
                class_data = X[class_mask]
                
                if len(class_data) < 10:  # Skip if too few samples
                    print(f"Skipping {class_name} - too few samples")
                    continue
                
                # Train GAN for this class
                gan = NetworkGAN(input_dim=X.shape[1])
                gan.train(class_data, epochs=500, batch_size=min(32, len(class_data)))
                
                # Generate synthetic samples
                samples_needed = target_samples - count
                synthetic_data = gan.generate_samples(samples_needed)
                
                # Scale synthetic data to match real data distribution
                synthetic_data = synthetic_data * (class_data.max(axis=0) - class_data.min(axis=0)) + class_data.min(axis=0)
                
                augmented_X.append(synthetic_data)
                augmented_y.append(np.full(samples_needed, class_idx))
                
                self.gan_models[class_name] = gan
        
        # Combine all data
        X_augmented = np.vstack(augmented_X)
        y_augmented = np.concatenate(augmented_y)
        
        print(f"Dataset augmented from {len(X)} to {len(X_augmented)} samples")
        return X_augmented, y_augmented
    
    def train_classifier(self, X_train, y_train):
        """Train the final classifier"""
        print("Training enhanced Random Forest classifier...")
        
        self.classifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=25,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        self.classifier.fit(X_train, y_train)
        return self.classifier
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model"""
        predictions = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"\n=== GAN-Enhanced IDS Results ===")
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, predictions, target_names=self.label_encoder.classes_))
        
        return accuracy, predictions
    
    def visualize_gan_results(self, X_original, X_augmented, y_augmented):
        """Visualize GAN augmentation results"""
        from sklearn.decomposition import PCA
        
        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_augmented)
        
        plt.figure(figsize=(12, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.label_encoder.classes_)))
        
        for i, class_name in enumerate(self.label_encoder.classes_):
            mask = y_augmented == i
            if np.any(mask):
                plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                           c=[colors[i]], label=class_name, alpha=0.6, s=20)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('GAN-Augmented Network Traffic Data - PCA Visualization')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('gan_augmented_data.png', dpi=300, bbox_inches='tight')
        print("✓ GAN augmentation visualization saved")

def main():
    """Main execution function"""
    print("=== GAN-Enhanced Network Intrusion Detection System ===")
    print("Using Generative Adversarial Networks for Data Augmentation\n")
    
    # Initialize system
    ids = GANEnhancedIDS()
    
    # Load and preprocess data
    df = ids.load_data()
    X, y, feature_names = ids.preprocess_data(df)
    
    print(f"Original dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Augment minority classes with GANs
    X_train_augmented, y_train_augmented = ids.augment_minority_classes(X_train, y_train)
    
    print(f"Augmented training set: {X_train_augmented.shape[0]} samples")
    print(f"New class distribution: {np.bincount(y_train_augmented)}")
    
    # Train classifier
    ids.train_classifier(X_train_augmented, y_train_augmented)
    
    # Evaluate model
    accuracy, predictions = ids.evaluate_model(X_test, y_test)
    
    # Visualize results
    ids.visualize_gan_results(X, X_train_augmented, y_train_augmented)
    
    # Save results
    results = {
        'original_samples': int(X.shape[0]),
        'augmented_samples': int(X_train_augmented.shape[0]),
        'test_accuracy': float(accuracy),
        'classes': list(ids.label_encoder.classes_),
        'gan_models_trained': list(ids.gan_models.keys())
    }
    
    with open('gan_enhanced_ids_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Summary ===")
    print(f"✓ Original dataset: {X.shape[0]} samples")
    print(f"✓ GAN-augmented dataset: {X_train_augmented.shape[0]} samples")
    print(f"✓ Test accuracy: {accuracy:.4f}")
    print(f"✓ GAN models trained for: {list(ids.gan_models.keys())}")
    print(f"✓ Results saved to 'gan_enhanced_ids_results.json'")
    
    return ids, results

if __name__ == "__main__":
    ids, results = main()