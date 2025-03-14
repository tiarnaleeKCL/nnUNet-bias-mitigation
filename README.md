# nnU-Net bias mitigation

We investigated the impact of common bias mitigation methods to address bias between Black and White subjects in AI-based CMR segmentation models. Specifically, we use oversampling, importance reweighing and Group DRO  as well as combinations of these techniques to mitigate the race bias. Furthermore, motivated by recent findings on the root causes of AI-based CMR segmentation bias [1], we evaluate the same methods using models trained and evaluated on cropped CMR images. We find that bias can be mitigated using oversampling, significantly improving performance for the underrepresented Black subjects whilst not significantly reducing the majority White subjects' performance. Group DRO also improves performance for Black subjects but not significantly, while reweighing decreases performance for Black subjects. Using a combination of oversampling and Group DRO also improves performance for Black subjects but not significantly. Using cropped images increases performance for both races and reduces the bias, whilst adding oversampling as a bias mitigation technique with cropped images reduces the bias further.

# Using the code
This code is based on [nnU-Net v1](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1). To use the bias mitigation models:
- Create a labels.csv file with the subject name and race label and save it in the folder raw_data_base/[your task name]/. E.g.

| Subject  | Race label |
| ------------- | ------------- |
| Subject 1 | 1 |
| Subject 2 | 2 |
| Subject 3 | 2 |
| Subject 4 | 1 |

The labels must be numeric and the subject name must match the format of the segmentation file names e.g. for a cardiac MR segmentation 1234567_sa_ED.nii.gz, the subject name should be 1234567_sa_ED 
- Add the dictionary key called 'race label' to the nnUNet_preprocessed/[your task_name]/nnUNetData_plans_v2.1_2D_stage0 pickle file
- For oversampling, replace `dl_tr` and `dl_val = DataLoader2D` with `dl_tr = DataLoader2D_oversampling` in `training/network_training/nnUNetTrainer`. Add the path to your labels.csv file to `dataloading/dataset_loading` --> `DataLoader2D_oversampling.generate_train_batch()`
- For GroupDRO and reweighing, replace `self.loss = DC_and_CE_loss` with the relevant loss function: `DC_and_CE_loss_with_reweighting` for reweighing, `GroupDRO_with_CE` for GroupDRO and `GroupDRO_with_CE_reweighting` for GroupDRO + reweighing in `training/network_training/nnUNetTrainerV2`


[1] Tiarna Lee, Esther Puyol-Antón, Bram Ruijsink, Sebastien Roujol, Theodore Barfoot, Shaheim Ogbomo-Harmitt, Miaojing Shi, Andrew King, An investigation into the causes of race bias in artificial intelligence–based cine cardiac magnetic resonance segmentation, European Heart Journal - Digital Health, 2025;, ztaf008, https://doi.org/10.1093/ehjdh/ztaf008
