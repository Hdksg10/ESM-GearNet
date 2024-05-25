import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import esm
from tensorboardX import SummaryWriter
from datetime import datetime


class ActivityPredictionModel(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(ActivityPredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class ProteinDataset(Dataset):
    def __init__(self, annotations_file, sequence, batch_converter, transform=None):
        self.annotations_frame = pd.read_csv(annotations_file)
        self.sequence = sequence
        self.transform = transform
        self.batch_converter=batch_converter

    def __len__(self):
        return len(self.annotations_frame)

    def __getitem__(self, idx):
        row = self.annotations_frame.iloc[idx]
        mutant = row['Mutant']
        # activity = row['Activity']
        # selectivity = row['Selectivity']

        # 应用突变到序列
        mutated_sequence = apply_mutations_to_sequence(self.sequence, mutant)

        # 使用 transform 变换序列
        if self.transform:
            mutated_sequence = self.transform(mutated_sequence)
        _, _, batch_tokens = self.batch_converter([(str(idx), mutated_sequence)])


        # 创建一个样本字典
        sample = {
            'data': batch_tokens,
            # 'activity': torch.tensor(activity, dtype=torch.float),
            # 'selectivity': torch.tensor(selectivity, dtype=torch.float)
        }

        return sample

# 应用突变的函数定义
def apply_mutations_to_sequence(sequence, mutations_str):
    if mutations_str == "WT" or not mutations_str:
        # 如果没有突变（即野生型），直接返回原始序列
        return sequence
    mutations = mutations_str.split(';')
    sequence = list(sequence)  # 转换成列表便于修改
    for mut in mutations:
        if mut:  # 确保突变不为空
            original_aa, position, new_aa = mut[0], int(mut[1:-1]) - 1, mut[-1]
            assert sequence[position] == original_aa, f"Mutation at position {position+1} does not match the original amino acid."
            sequence[position] = new_aa
    return ''.join(sequence)


# Sample tokenizer function
def simple_tokenizer(sequence):
    # 直接返回序列，不做任何转换
    return sequence
# Protein sequence as provided
if __name__ == "__main__":
    protein_sequence = "MRRESLLVSVCKGLRVHVERVGQDPGRSTVMLVNGAMATTASFARTCKCLAEHFNVVLFDLPFAGQSRQHNPQRGLITKDDEVEILLALIERFEVNHLVSASWGGISTLLALSRNPRGIRSSVVMAFAPGLNQAMLDYVGRAQALIELDDKSAIGHLLNETVGKYLPQRLKASNHQHMASLATGEYEQARFHIDQVLALNDRGYLACLERIQSHVHFINGSWDEYTTAEDARQFRDYLPHCSFSRVEGTGHFLDLESKLAAVRVHRALLEHLLKQPEPQRAERAAGFHEMAIGYA"
    device = "cuda:3"
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()  # disables dropout for deterministic results
    # Instantiate the dataset
    protein_dataset = ProteinDataset('/root/project/protein/esm/task_AI4S/data/train.csv', protein_sequence, batch_converter)
    train_size = int(0.8 * len(protein_dataset))
    val_size = len(protein_dataset)-train_size
    train_dataset, val_dataset = random_split(protein_dataset, [train_size, val_size])
    dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    # Create a DataLoader
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    model_pre = ActivityPredictionModel(input_dim=1280).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_pre.parameters(), lr=0.001)
    model_save_path = './madel_save'
    os.makedirs(model_save_path, exist_ok=True)
    best_loss = float('inf') 
    current_time = datetime.now().strftime('%m-%d_%H-%M-%S')
    writer = SummaryWriter(f"./runs/pre_{current_time}")
    # Check a sample from the dataloader
    num_epochs = 50

    for epoch in range(num_epochs):
        total_loss = 0
        total_samples = 0
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()  # 梯度归零
        
            # 将数据移到 GPU 上
            data_tuples = batch['data'].squeeze().to(device)
            selectivity = batch['selectivity'].to(device)
            
            with torch.no_grad():
                results = model(data_tuples, repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33].to(device)  # 移动到 GPU
            mean_representations = token_representations.mean(dim=1)
            predictions = model_pre(mean_representations)
            loss = criterion(predictions.squeeze(), selectivity)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_samples += selectivity.size(0)
            
            writer.add_scalar(f'Loss/train_batch/num_epochs_{epoch}', loss.item(), batch_idx)

        avg_loss = total_loss / total_samples
        writer.add_scalar('Loss/Train', avg_loss, epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            # 保存模型
            torch.save(model_pre.state_dict(), os.path.join(model_save_path, 'selectivity_best_model.pth'))

    torch.save(model_pre.state_dict(), os.path.join(model_save_path, 'selectivity_final_model.pth'))
    print("Final model saved!")


### 验证集得出结果
