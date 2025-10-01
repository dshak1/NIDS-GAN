"""
GAN-Based Network Traffic Data Generator for Intrusion Detection
Generates synthetic network traffic data to augment training datasets
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns

class NetworkTrafficGAN:
    def __init__(self, input_dim, noise_dim=100):
        self.input_dim = input_dim
        self.noise_dim = noise_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()
        
    def build_generator(self):
        """Build generator network to create synthetic network traffic data"""
        model = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_dim=self.noise_dim),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(self.input_dim, activation='tanh')  # tanh for normalized data
        ])
        
        model.compile(optimizer='adam')
        return model
    
    def build_discriminator(self):
        """Build discriminator network to distinguish real vs synthetic data"""
        model = tf.keras.Sequential([
            layers.Dense(512, activation='leaky_relu', input_dim=self.input_dim),
            layers.Dropout(0.3),
            
            layers.Dense(256, activation='leaky_relu'),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='leaky_relu'),
            layers.Dropout(0.3),
            
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def build_gan(self):
        """Build combined GAN model"""
        self.discriminator.trainable = False
        
        model = tf.keras.Sequential([
            self.generator,
            self.discriminator
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model
    
    def train(self, X_real, epochs=10000, batch_size=32, save_interval=1000):
        """Train the GAN on real network traffic data"""
        
        # Normalize real data to [-1, 1] for tanh activation
        X_real = (X_real - X_real.min()) / (X_real.max() - X_real.min()) * 2 - 1
        
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        d_losses = []
        g_losses = []
        
        for epoch in range(epochs):
            # Train Discriminator
            idx = np.random.randint(0, X_real.shape[0], batch_size)
            real_data = X_real[idx]
            
            # Generate fake data
            noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
            fake_data = self.generator.predict(noise, verbose=0)
            
            # Train discriminator on real and fake data
            d_loss_real = self.discriminator.train_on_batch(real_data, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_data, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train Generator
            noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
            g_loss = self.gan.train_on_batch(noise, valid)
            
            d_losses.append(d_loss[0])
            g_losses.append(g_loss)
            
            if epoch % save_interval == 0:
                print(f"Epoch {epoch}: D Loss: {d_loss[0]:.4f}, G Loss: {g_loss:.4f}")
                
        return d_losses, g_losses
    
    def generate_synthetic_data(self, num_samples, attack_type='dos'):
        """Generate synthetic network traffic data"""
        noise = np.random.normal(0, 1, (num_samples, self.noise_dim))
        synthetic_data = self.generator.predict(noise, verbose=0)
        
        # Denormalize data back to original scale
        synthetic_data = (synthetic_data + 1) / 2  # Convert from [-1,1] to [0,1]
        
        return synthetic_data

class ModernNetworkDataGenerator:
    """Generate modern network traffic patterns using domain knowledge"""
    
    def __init__(self):
        self.feature_names = [
            'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
            'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
            'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'count', 'srv_count',
            'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate'
        ]
    
    def generate_ddos_attack(self, num_samples=1000):
        """Generate modern DDoS attack patterns"""
        data = []
        
        for _ in range(num_samples):
            record = {
                'duration': np.random.exponential(0.1),  # Very short connections
                'src_bytes': np.random.exponential(100),   # Small packets
                'dst_bytes': 0,  # No response typical in DDoS
                'land': 0,
                'wrong_fragment': np.random.binomial(1, 0.1),
                'urgent': 0,
                'hot': 0,
                'num_failed_logins': 0,
                'logged_in': 0,
                'num_compromised': 0,
                'root_shell': 0,
                'su_attempted': 0,
                'num_root': 0,
                'num_file_creations': 0,
                'num_shells': 0,
                'num_access_files': 0,
                'count': np.random.poisson(500),  # High connection count
                'srv_count': np.random.poisson(500),
                'serror_rate': np.random.beta(5, 2),  # High error rate
                'srv_serror_rate': np.random.beta(5, 2),
                'rerror_rate': np.random.beta(2, 5),
                'srv_rerror_rate': np.random.beta(2, 5),
                'same_srv_rate': np.random.beta(8, 2),  # Same service targeted
                'diff_srv_rate': np.random.beta(1, 8),
                'srv_diff_host_rate': np.random.beta(1, 8),
                'dst_host_count': np.random.poisson(255),  # High host count
                'dst_host_srv_count': np.random.poisson(255),
                'dst_host_same_srv_rate': np.random.beta(8, 2),
                'dst_host_diff_srv_rate': np.random.beta(1, 8),
                'dst_host_same_src_port_rate': np.random.beta(1, 8),
                'attack_type': 'dos'
            }
            data.append(record)
        
        return pd.DataFrame(data)
    
    def generate_port_scan(self, num_samples=1000):
        """Generate modern port scanning attack patterns"""
        data = []
        
        for _ in range(num_samples):
            record = {
                'duration': 0,  # Connection attempts, no established connection
                'src_bytes': np.random.poisson(50),   # Small probe packets
                'dst_bytes': 0,  # No response or RST packets
                'land': 0,
                'wrong_fragment': 0,
                'urgent': 0,
                'hot': 0,
                'num_failed_logins': 0,
                'logged_in': 0,
                'num_compromised': 0,
                'root_shell': 0,
                'su_attempted': 0,
                'num_root': 0,
                'num_file_creations': 0,
                'num_shells': 0,
                'num_access_files': 0,
                'count': np.random.poisson(100),  # Many connection attempts
                'srv_count': np.random.poisson(20),  # Different services
                'serror_rate': np.random.beta(8, 2),  # High error rate (rejections)
                'srv_serror_rate': np.random.beta(8, 2),
                'rerror_rate': np.random.beta(2, 8),
                'srv_rerror_rate': np.random.beta(2, 8),
                'same_srv_rate': np.random.beta(1, 8),  # Different services
                'diff_srv_rate': np.random.beta(8, 2),  # Scanning different services
                'srv_diff_host_rate': np.random.beta(1, 8),
                'dst_host_count': 1,  # Single target host
                'dst_host_srv_count': np.random.poisson(50),  # Many services on target
                'dst_host_same_srv_rate': np.random.beta(1, 8),
                'dst_host_diff_srv_rate': np.random.beta(8, 2),
                'dst_host_same_src_port_rate': np.random.beta(2, 8),
                'attack_type': 'probe'
            }
            data.append(record)
        
        return pd.DataFrame(data)
    
    def generate_normal_traffic(self, num_samples=1000):
        """Generate normal network traffic patterns"""
        data = []
        
        for _ in range(num_samples):
            record = {
                'duration': np.random.exponential(30),  # Normal session duration
                'src_bytes': np.random.lognormal(8, 2),   # Varied data sizes
                'dst_bytes': np.random.lognormal(7, 2),   # Response data
                'land': 0,
                'wrong_fragment': np.random.binomial(1, 0.01),  # Rare fragmentation
                'urgent': np.random.binomial(1, 0.01),
                'hot': np.random.binomial(1, 0.05),
                'num_failed_logins': np.random.binomial(3, 0.1),
                'logged_in': np.random.binomial(1, 0.7),  # Usually logged in
                'num_compromised': 0,
                'root_shell': 0,
                'su_attempted': 0,
                'num_root': 0,
                'num_file_creations': np.random.poisson(2),
                'num_shells': 0,
                'num_access_files': np.random.poisson(5),
                'count': np.random.poisson(10),  # Normal connection count
                'srv_count': np.random.poisson(5),
                'serror_rate': np.random.beta(1, 20),  # Low error rate
                'srv_serror_rate': np.random.beta(1, 20),
                'rerror_rate': np.random.beta(1, 20),
                'srv_rerror_rate': np.random.beta(1, 20),
                'same_srv_rate': np.random.beta(5, 5),  # Mixed service usage
                'diff_srv_rate': np.random.beta(3, 7),
                'srv_diff_host_rate': np.random.beta(2, 8),
                'dst_host_count': np.random.poisson(20),
                'dst_host_srv_count': np.random.poisson(10),
                'dst_host_same_srv_rate': np.random.beta(3, 7),
                'dst_host_diff_srv_rate': np.random.beta(7, 3),
                'dst_host_same_src_port_rate': np.random.beta(5, 5),
                'attack_type': 'normal'
            }
            data.append(record)
        
        return pd.DataFrame(data)

def main():
    """Demonstrate GAN-based data generation for network intrusion detection"""
    
    print("=== Modern Network Traffic Data Generation ===\n")
    
    # Generate modern synthetic data using domain knowledge
    generator = ModernNetworkDataGenerator()
    
    print("Generating synthetic network traffic data...")
    normal_data = generator.generate_normal_traffic(5000)
    ddos_data = generator.generate_ddos_attack(2000)
    scan_data = generator.generate_port_scan(1000)
    
    # Combine all data
    synthetic_df = pd.concat([normal_data, ddos_data, scan_data], ignore_index=True)
    
    print(f"Generated {len(synthetic_df)} synthetic network records")
    print(f"Attack distribution:\n{synthetic_df['attack_type'].value_counts()}")
    
    # Prepare data for GAN training (using a subset of normal traffic)
    normal_features = normal_data.drop('attack_type', axis=1)
    
    print("\n=== Training GAN on Normal Traffic ===")
    
    # Initialize and train GAN
    gan = NetworkTrafficGAN(input_dim=normal_features.shape[1])
    
    # Train GAN (reduced epochs for demo)
    d_losses, g_losses = gan.train(normal_features.values, epochs=1000, batch_size=32, save_interval=200)
    
    # Generate additional synthetic normal data
    additional_normal = gan.generate_synthetic_data(1000)
    
    print(f"\nGenerated {len(additional_normal)} additional synthetic normal traffic records using GAN")
    
    # Save the synthetic dataset
    synthetic_df.to_csv('synthetic_network_data.csv', index=False)
    print("\nSynthetic dataset saved as 'synthetic_network_data.csv'")
    
    return synthetic_df

if __name__ == "__main__":
    main()