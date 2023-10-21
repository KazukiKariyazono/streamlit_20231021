import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, RDKFingerprint
from rdkit.Chem import Draw
import torch
import torch.nn as nn

# ネットワークの定義
class Net(nn.Module):
    def __init__(self, n_feature, hidden_dim, n_output, n_layers):
        super().__init__()
        self.n_feature = n_feature
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.input_layer = nn.Sequential(
            nn.BatchNorm1d(n_feature),
            nn.Linear(n_feature, hidden_dim),
            nn.ReLU()
        )

        middle = []
        for _ in range(n_layers):
            middle.append(nn.BatchNorm1d(hidden_dim))
            middle.append(nn.Linear(hidden_dim, hidden_dim))
            middle.append(nn.ReLU())

        self.middle_layers = nn.Sequential(*middle)
        self.output_layer = nn.Linear(hidden_dim, n_output)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.middle_layers(x)
        x = self.output_layer(x)
        return x

# 重みファイルのロード
n_feature = 2048 # 特徴の数を設定
hidden_dim = 75 # 隠れ層の次元を設定
n_layers = 12 # 中間層の数を設定
n_output = 1 # 出力層の次元を設定

model = Net(n_feature=n_feature, hidden_dim=hidden_dim, n_layers=n_layers, n_output=n_output)
model.load_state_dict(torch.load("weights_231008.pth"))
model.eval()

# テキストボックスとスライダーの設定
monomer1 = st.sidebar.text_input("Monomer 1 (SMILES)")
slider1 = st.sidebar.slider("Monomer 1 Percentage", 0, 100, 100, 1)  # 初期値 100, ステップ 1
monomer2 = st.sidebar.text_input("Monomer 2 (SMILES)")
slider2 = st.sidebar.slider("Monomer 2 Percentage", 0, 100, 0, 1)  # 初期値 0, ステップ 1
monomer3 = st.sidebar.text_input("Monomer 3 (SMILES)")
slider3 = st.sidebar.slider("Monomer 3 Percentage", 0, 100, 0, 1)  # 初期値 0, ステップ 1

# 合計パーセンテージの計算
total_percentage = slider1 + slider2 + slider3

# SMILESから分子構造への変換
def smiles_to_molecule(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    return molecule

# RDKFingerprintの計算
def calculate_fingerprint(molecule):
    fingerprint = RDKFingerprint(molecule)
    return fingerprint

# 屈折率の推定
def predict_refractive_index(fingerprint, percentage):
    # 2D Fingerprintをモデルに入力
    with torch.no_grad():
        # Check fingerprint type and convert to PyTorch tensor if it's not
        if not isinstance(fingerprint, torch.Tensor):
            if isinstance(fingerprint, Chem.RWMol):
                fingerprint = Chem.RDKFingerprint(fingerprint)  # RDKitのBitVectに変換
            fingerprint = torch.tensor(fingerprint, dtype=torch.float32)  # PyTorchテンソルに変換
        # Check fingerprint dimension and add a batch dimension if it's 1D
        if fingerprint.dim() == 1:
            fingerprint = fingerprint.unsqueeze(0)  # バッチ次元を追加
        prediction = model(fingerprint)
    return prediction.item() * percentage / 100

# メインコンテンツ
st.title("屈折率推算アプリ")

# 分子構造とスライダーの表示
with st.expander("Input Value"):
    molecules = []
    percentages = [slider1, slider2, slider3]
    for i, monomer in enumerate([monomer1, monomer2, monomer3]):
        if monomer:
            molecule = smiles_to_molecule(monomer)
            if molecule:  # SMILESが有効な分子に変換できた場合のみ処理
                molecules.append(molecule)
                mol_img = Draw.MolToImage(molecule, size=(100, 100))
                st.image(mol_img, caption=f"Monomer {i + 1}")
                st.write(f"Monomer {i + 1} Percentage: {percentages[i]}%")

# 推定屈折率の表示
with st.expander("Prediction"):
    refractive_indices = []
    for i, molecule in enumerate(molecules):
        if not molecule:  # SMILES が未記入の場合、寄与をゼロにする
            adjusted_percentage = 0
        else:
            adjusted_percentage = percentages[i]
            fingerprint = calculate_fingerprint(molecule)
        
        # 100％の屈折率を表示
        refractive_index = predict_refractive_index(fingerprint, 100)
        refractive_indices.append(refractive_index)
        
        st.write(f"Monomer {i + 1} Estimated Refractive Index: {refractive_index:.4f} (without percentage)")

# 平均屈折率の計算と表示
with st.expander("Result"):
    valid_molecules = [molecule for molecule in molecules if molecule is not None]
    valid_percentages = [percentage for percentage, molecule in zip(percentages, molecules) if molecule is not None]  # SMILES が未記入の場合のパーセンテージを考慮
    if len(valid_molecules) > 0:
        valid_refractive_indices = [predict_refractive_index(calculate_fingerprint(molecule), percentage) for molecule, percentage in zip(valid_molecules, valid_percentages)]  # 各モノマーのパーセンテージを考慮
        average_refractive_index = sum(valid_refractive_indices)
        st.write(f"This material's estimated refractive index is approximately: {average_refractive_index:.4f}")
    else:
        st.write("No valid molecules found for refractive index calculation.")

# 合計の表示
st.write("Total Percentage:", total_percentage)
# 合計が100でない場合にメッセージを表示
if total_percentage != 100:
    st.write("合計を100にしてください")