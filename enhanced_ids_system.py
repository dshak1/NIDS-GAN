"""
Enhanced Network Intrusion Detection System
Uses real-world cybersecurity datasets and advanced ML techniques
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import json
import warnings
warnings.filterwarnings('ignore')

class EnhancedIntrusionDetector:
    """Enhanced Network Intrusion Detection System with modern techniques"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.feature_importance = None
        
    def load_real_world_data(self):
        """Load the real-world datasets we generated"""
        print("Loading real-world cybersecurity datasets...")
        
        try:
            # Load network traffic data
            network_data = pd.read_csv('enriched_network_traffic.csv')
            print(f"Loaded {len(network_data)} network traffic records")
            
            # Load endpoint security data
            endpoint_data = pd.read_csv('endpoint_security_events.csv')
            print(f"Loaded {len(endpoint_data)} endpoint security events")
            
            # Load threat intelligence
            with open('threat_intelligence.json', 'r') as f:
                threat_intel = json.load(f)
            
            return network_data, endpoint_data, threat_intel
            
        except FileNotFoundError:
            print("Real-world data files not found. Generating synthetic data...")
            return self.generate_enhanced_synthetic_data()
    
    def generate_enhanced_synthetic_data(self):
        """Generate enhanced synthetic data with modern attack patterns"""
        print("Generating enhanced synthetic cybersecurity data...")
        
        np.random.seed(42)  # For reproducibility
        
        # Define modern attack types and their characteristics
        attack_profiles = {
            'BENIGN': {
                'samples': 10000,
                'duration_range': (1, 300),
                'packet_count_range': (10, 200),
                'bytes_range': (1000, 100000),
                'packets_per_sec_range': (1, 50),
                'error_rate': 0.02
            },
            'DDoS': {
                'samples': 2000,
                'duration_range': (0.1, 5),
                'packet_count_range': (1000, 10000),
                'bytes_range': (64000, 640000),
                'packets_per_sec_range': (1000, 50000),
                'error_rate': 0.8
            },
            'Port Scan': {
                'samples': 1500,
                'duration_range': (0.01, 1),
                'packet_count_range': (1, 10),
                'bytes_range': (40, 400),
                'packets_per_sec_range': (10, 1000),
                'error_rate': 0.9
            },
            'Brute Force': {
                'samples': 1000,
                'duration_range': (30, 600),
                'packet_count_range': (50, 500),
                'bytes_range': (5000, 50000),
                'packets_per_sec_range': (1, 10),
                'error_rate': 0.7
            },
            'Botnet': {
                'samples': 800,
                'duration_range': (60, 3600),
                'packet_count_range': (20, 100),
                'bytes_range': (2000, 20000),
                'packets_per_sec_range': (0.1, 2),
                'error_rate': 0.1
            },
            'Web Attack': {
                'samples': 1200,
                'duration_range': (1, 30),
                'packet_count_range': (5, 100),
                'bytes_range': (500, 50000),
                'packets_per_sec_range': (1, 50),
                'error_rate': 0.3
            }
        }
        
        all_data = []
        
        for attack_type, profile in attack_profiles.items():
            print(f"Generating {profile['samples']} samples for {attack_type}...")
            
            for _ in range(profile['samples']):
                # Basic flow features
                duration = np.random.uniform(*profile['duration_range'])
                packet_count = np.random.randint(*profile['packet_count_range'])
                total_bytes = np.random.randint(*profile['bytes_range'])
                packets_per_sec = min(packet_count / max(duration, 0.001), 
                                    np.random.uniform(*profile['packets_per_sec_range']))
                
                # Advanced features
                avg_packet_size = total_bytes / max(packet_count, 1)
                bytes_per_sec = total_bytes / max(duration, 0.001)
                
                # Protocol flags (TCP flags simulation)
                syn_count = np.random.poisson(2) if attack_type != 'DDoS' else np.random.poisson(packet_count/100)
                ack_count = packet_count - syn_count if attack_type == 'BENIGN' else np.random.poisson(packet_count/10)
                fin_count = np.random.poisson(1) if attack_type == 'BENIGN' else 0
                rst_count = np.random.poisson(packet_count/10) if 'Scan' in attack_type else np.random.poisson(0.5)
                
                # Error rates
                error_rate = np.random.beta(1, 20) if attack_type == 'BENIGN' else np.random.beta(5, 2)
                
                # Flow statistics
                flow_iat_mean = duration / max(packet_count, 1) * 1000  # Inter-arrival time in ms
                flow_iat_std = flow_iat_mean * np.random.uniform(0.1, 2)
                
                # Advanced network features
                fwd_packets = int(packet_count * np.random.uniform(0.4, 0.8))
                bwd_packets = packet_count - fwd_packets
                
                fwd_bytes = int(total_bytes * np.random.uniform(0.3, 0.9))
                bwd_bytes = total_bytes - fwd_bytes
                
                # Subflow features
                subflow_fwd_packets = fwd_packets
                subflow_bwd_packets = bwd_packets
                subflow_fwd_bytes = fwd_bytes
                subflow_bwd_bytes = bwd_bytes
                
                # Header lengths
                fwd_header_len = fwd_packets * 20  # Assume 20 bytes per header
                bwd_header_len = bwd_packets * 20
                
                record = {
                    # Basic flow features
                    'flow_duration': duration,
                    'total_fwd_packets': fwd_packets,
                    'total_backward_packets': bwd_packets,
                    'total_length_fwd_packets': fwd_bytes,
                    'total_length_bwd_packets': bwd_bytes,
                    
                    # Packet size features
                    'fwd_packet_length_max': avg_packet_size * np.random.uniform(1.5, 3),
                    'fwd_packet_length_min': avg_packet_size * np.random.uniform(0.1, 0.5),
                    'fwd_packet_length_mean': fwd_bytes / max(fwd_packets, 1),
                    'fwd_packet_length_std': avg_packet_size * np.random.uniform(0.1, 1),
                    
                    'bwd_packet_length_max': avg_packet_size * np.random.uniform(1.2, 2.5),
                    'bwd_packet_length_min': avg_packet_size * np.random.uniform(0.1, 0.4),
                    'bwd_packet_length_mean': bwd_bytes / max(bwd_packets, 1),
                    'bwd_packet_length_std': avg_packet_size * np.random.uniform(0.1, 0.8),
                    
                    # Flow statistics
                    'flow_bytes_s': bytes_per_sec,
                    'flow_packets_s': packets_per_sec,
                    'flow_iat_mean': flow_iat_mean,
                    'flow_iat_std': flow_iat_std,
                    'flow_iat_max': flow_iat_mean * np.random.uniform(2, 10),
                    'flow_iat_min': flow_iat_mean * np.random.uniform(0.1, 0.5),
                    
                    # Forward/Backward IAT
                    'fwd_iat_total': flow_iat_mean * fwd_packets,
                    'fwd_iat_mean': flow_iat_mean * np.random.uniform(0.8, 1.2),
                    'fwd_iat_std': flow_iat_std * np.random.uniform(0.5, 1.5),
                    'fwd_iat_max': flow_iat_mean * np.random.uniform(3, 8),
                    'fwd_iat_min': flow_iat_mean * np.random.uniform(0.1, 0.3),
                    
                    'bwd_iat_total': flow_iat_mean * bwd_packets,
                    'bwd_iat_mean': flow_iat_mean * np.random.uniform(0.9, 1.1),
                    'bwd_iat_std': flow_iat_std * np.random.uniform(0.4, 1.3),
                    'bwd_iat_max': flow_iat_mean * np.random.uniform(2, 6),
                    'bwd_iat_min': flow_iat_mean * np.random.uniform(0.05, 0.2),
                    
                    # Flags
                    'fwd_psh_flags': np.random.poisson(fwd_packets * 0.1),
                    'bwd_psh_flags': np.random.poisson(bwd_packets * 0.1),
                    'fwd_urg_flags': np.random.poisson(fwd_packets * 0.01),
                    'bwd_urg_flags': np.random.poisson(bwd_packets * 0.01),
                    'fwd_header_length': fwd_header_len,
                    'bwd_header_length': bwd_header_len,
                    
                    # Packets per second
                    'fwd_packets_s': fwd_packets / max(duration, 0.001),
                    'bwd_packets_s': bwd_packets / max(duration, 0.001),
                    
                    # Packet length statistics
                    'min_packet_length': avg_packet_size * np.random.uniform(0.1, 0.3),
                    'max_packet_length': avg_packet_size * np.random.uniform(2, 4),
                    'packet_length_mean': avg_packet_size,
                    'packet_length_std': avg_packet_size * np.random.uniform(0.2, 1),
                    'packet_length_variance': (avg_packet_size * np.random.uniform(0.2, 1)) ** 2,
                    
                    # TCP Flags
                    'fin_flag_count': fin_count,
                    'syn_flag_count': syn_count,
                    'rst_flag_count': rst_count,
                    'psh_flag_count': np.random.poisson(packet_count * 0.1),
                    'ack_flag_count': ack_count,
                    'urg_flag_count': np.random.poisson(packet_count * 0.01),
                    'cwe_flag_count': np.random.poisson(packet_count * 0.01),
                    'ece_flag_count': np.random.poisson(packet_count * 0.01),
                    
                    # Ratios
                    'down_up_ratio': bwd_bytes / max(fwd_bytes, 1),
                    'average_packet_size': avg_packet_size,
                    'avg_fwd_segment_size': fwd_bytes / max(fwd_packets, 1),
                    'avg_bwd_segment_size': bwd_bytes / max(bwd_packets, 1),
                    
                    # Subflow features
                    'subflow_fwd_packets': subflow_fwd_packets,
                    'subflow_fwd_bytes': subflow_fwd_bytes,
                    'subflow_bwd_packets': subflow_bwd_packets,
                    'subflow_bwd_bytes': subflow_bwd_bytes,
                    
                    # Init features
                    'init_win_bytes_forward': np.random.randint(1024, 65536),
                    'init_win_bytes_backward': np.random.randint(1024, 65536),
                    'act_data_pkt_fwd': fwd_packets,
                    'min_seg_size_forward': np.random.randint(20, 100),
                    
                    # Active/Idle times
                    'active_mean': duration * np.random.uniform(0.6, 0.9),
                    'active_std': duration * np.random.uniform(0.1, 0.3),
                    'active_max': duration * np.random.uniform(0.8, 1.0),
                    'active_min': duration * np.random.uniform(0.1, 0.4),
                    
                    'idle_mean': duration * np.random.uniform(0.1, 0.3),
                    'idle_std': duration * np.random.uniform(0.05, 0.2),
                    'idle_max': duration * np.random.uniform(0.2, 0.5),
                    'idle_min': duration * np.random.uniform(0.01, 0.1),
                    
                    # Label
                    'label': attack_type
                }
                
                all_data.append(record)
        
        df = pd.DataFrame(all_data)
        print(f"Generated {len(df)} total records")
        print(f"Label distribution:\n{df['label'].value_counts()}")
        
        return df, None, None  # Return None for endpoint_data and threat_intel
    
    def preprocess_data(self, df):
        """Preprocess the data for machine learning"""
        print("Preprocessing data...")
        
        # Handle datetime columns if they exist
        datetime_columns = []
        for col in df.columns:
            if df[col].dtype == 'object' and col != 'label':
                try:
                    # Try to convert to datetime
                    pd.to_datetime(df[col].iloc[0])
                    datetime_columns.append(col)
                except:
                    pass
        
        # Drop datetime columns and other non-numeric columns
        columns_to_drop = datetime_columns + ['src_ip', 'dst_ip', 'service', 'protocol', 
                                             'tcp_flags', 'src_port', 'hostname', 
                                             'event_type', 'process_name', 'user', 
                                             'severity', 'file_hash', 'parent_process', 
                                             'command_line', 'src_country', 'dst_country']
        
        # Only drop columns that actually exist in the dataframe
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        
        if columns_to_drop:
            print(f"Dropping non-numeric columns: {columns_to_drop}")
            df = df.drop(columns=columns_to_drop)
        
        # Handle missing values and infinities
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        # Separate features and labels
        X = df.drop('label', axis=1)
        y = df['label']
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        
        print(f"Preprocessed data shape: {X_scaled_df.shape}")
        print(f"Classes: {list(self.label_encoder.classes_)}")
        print(f"Class distribution: {np.bincount(y_encoded)}")
        
        return X_scaled_df, y_encoded, X.columns
    
    def train_advanced_models(self, X_train, y_train):
        """Train advanced machine learning models"""
        print("\nTraining advanced ML models...")
        
        # Enhanced Random Forest
        self.models['Enhanced Random Forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Handle class imbalance
        )
        
        # Gradient Boosting
        self.models['Gradient Boosting'] = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=8,
            random_state=42
        )
        
        # Isolation Forest for anomaly detection
        self.models['Isolation Forest'] = IsolationForest(
            contamination=0.15,  # Expect 15% anomalies
            random_state=42,
            n_jobs=-1
        )
        
        results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            if name == 'Isolation Forest':
                # Train on normal data only
                normal_labels = ['BENIGN', 'Normal']  # Handle both labels
                normal_label = None
                for label in normal_labels:
                    if label in self.label_encoder.classes_:
                        normal_label = label
                        break
                
                if normal_label:
                    benign_mask = y_train == self.label_encoder.transform([normal_label])[0]
                    model.fit(X_train[benign_mask])
                    results[name] = "Anomaly Detection Model"
                else:
                    print(f"Warning: No normal class found in {self.label_encoder.classes_}")
                    continue
            else:
                model.fit(X_train, y_train)
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                results[name] = f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
        
        return results
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate the trained models"""
        print("\n=== Model Evaluation ===")
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n--- {name} ---")
            
            if name == 'Isolation Forest':
                # Anomaly detection evaluation
                predictions = model.predict(X_test)
                # Convert: -1 (anomaly) -> attack, 1 (normal) -> benign
                predictions = np.where(predictions == -1, 1, 0)
                
                # Binary classification: Normal vs others
                normal_labels = ['BENIGN', 'Normal']
                normal_label = None
                for label in normal_labels:
                    if label in self.label_encoder.classes_:
                        normal_label = label
                        break
                
                if normal_label:
                    y_binary = np.where(y_test == self.label_encoder.transform([normal_label])[0], 0, 1)
                else:
                    y_binary = np.ones(len(y_test))  # All attacks if no normal class
                
                accuracy = accuracy_score(y_binary, predictions)
                print(f"Anomaly Detection Accuracy: {accuracy:.4f}")
                
                results[name] = {'accuracy': accuracy, 'type': 'anomaly_detection'}
                
            else:
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                
                print(f"Accuracy: {accuracy:.4f}")
                
                # Detailed classification report
                class_names = self.label_encoder.classes_
                print("\nClassification Report:")
                print(classification_report(y_test, predictions, target_names=class_names))
                
                results[name] = {'accuracy': accuracy, 'predictions': predictions}
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance = model.feature_importances_
        
        return results
    
    def visualize_results(self, X, y, feature_names):
        """Create visualizations"""
        print("\nGenerating visualizations...")
        
        # Feature importance
        if self.feature_importance is not None:
            plt.figure(figsize=(12, 8))
            n_features = min(15, len(self.feature_importance))
            indices = np.argsort(self.feature_importance)[::-1][:n_features]
            
            plt.title(f"Top {n_features} Most Important Features")
            plt.bar(range(n_features), self.feature_importance[indices])
            plt.xticks(range(n_features), [feature_names[i] for i in indices], rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('enhanced_feature_importance.png', dpi=300, bbox_inches='tight')
            print("✓ Feature importance plot saved")
        
        # PCA visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        plt.figure(figsize=(12, 8))
        unique_labels = np.unique(y)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = y == label
            class_name = self.label_encoder.classes_[label]
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=[colors[i]], label=class_name, alpha=0.6, s=20)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('Network Traffic Data - PCA Visualization')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('enhanced_pca_visualization.png', dpi=300, bbox_inches='tight')
        print("✓ PCA visualization saved")
        
        # Class distribution
        plt.figure(figsize=(10, 6))
        class_counts = np.bincount(y)
        class_names = [self.label_encoder.classes_[i] for i in range(len(class_counts))]
        
        plt.bar(class_names, class_counts)
        plt.title('Attack Type Distribution')
        plt.xlabel('Attack Type')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
        print("✓ Class distribution plot saved")

def main():
    """Main execution function"""
    print("=== Enhanced Network Intrusion Detection System ===")
    print("Using Advanced ML Techniques and Real-World Data Patterns\n")
    
    # Initialize detector
    detector = EnhancedIntrusionDetector()
    
    # Load or generate data
    try:
        network_data, endpoint_data, threat_intel = detector.load_real_world_data()
        df = network_data
        print("Using real-world generated data")
    except:
        df, _, _ = detector.generate_enhanced_synthetic_data()
        print("Using enhanced synthetic data")
    
    # Preprocess data
    X, y, feature_names = detector.preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train models
    training_results = detector.train_advanced_models(X_train, y_train)
    for model_name, result in training_results.items():
        print(f"{model_name}: {result}")
    
    # Evaluate models
    evaluation_results = detector.evaluate_models(X_test, y_test)
    
    # Generate visualizations
    detector.visualize_results(X, y, feature_names)
    
    # Save results
    results_summary = {
        'dataset_info': {
            'total_samples': len(df),
            'features': len(feature_names),
            'classes': list(detector.label_encoder.classes_),
            'class_distribution': df['label'].value_counts().to_dict()
        },
        'model_performance': {
            name: result['accuracy'] if isinstance(result, dict) else str(result)
            for name, result in evaluation_results.items()
        }
    }
    
    with open('enhanced_ids_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n=== Enhanced IDS Summary ===")
    print(f"✓ Processed {len(df)} network traffic records")
    print(f"✓ Trained {len(detector.models)} advanced ML models")
    print(f"✓ Best model accuracies:")
    for name, result in evaluation_results.items():
        if isinstance(result, dict) and 'accuracy' in result:
            print(f"  - {name}: {result['accuracy']:.4f}")
    print(f"✓ Results saved to 'enhanced_ids_results.json'")
    print(f"✓ Visualizations saved as PNG files")
    
    return detector, evaluation_results

if __name__ == "__main__":
    detector, results = main()