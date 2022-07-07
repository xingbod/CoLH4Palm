# Co-Learning to Hash Palm Biometrics for Flexible IoT Deployment

Security enhancement via trustworthy identity authentication in Internet-of-Things (IoT) has soared recently. Biometrics offers a promising remedy to improve the security and utility of IoT and play a role in securing a variety of low-power and limited computing capability IoT devices to address identity management challenges. This paper proposes an IoT-compliant co-learned biometric hashing network derived from palm print and palm vein dubbed PalmCohashNet. The PalmCohashNet comprises two hashing sub-networks, one for each palm modality, and is trained collaboratively to generate shared hash codes for respective modality (co-hash codes). A cross-modality hashing loss is devised to encourage co-hash codes of palm vein and palm print from the same identity to be adjacent and consistent; meanwhile, pull the co-hash codes of each identity to a pre-assigned identity-specific hash centroid that is shared by both palm modalities. Two palm-based co-hash codes of a person can be generated simultaneously for deployment. The binary co-hash code is IoT compliant attributed to its highly compact form for storage and fast matching. A trained PalmCohashNet can be flexibly deployed under four operation modes: single-modality matching (print vs. print or vein vs. vein), multi-modality matching where both print and vein are utilized, and cross-modality matching (print vs. vein) depending on the IoT service context. Our empirical results on four publicly available palm databases show that the proposed method consistently outperforms state-of-the-art methods. 

# Contributions
- A novel palm print and palm vein co-learning hashing and flexible deployment framework is proposed to address the challenges of IoT-compliant biometrics-related services.
- We propose the PalmCohashNet to substantiate the very idea of the co-learning hashing notion. Furthermore, a cross-modality hashing loss is devised to minimize the modality gap between palm vein and palm print and encourage the model to produce discriminative, stable, compact, and low Hamming space utilization co-hash codes. 
- We conduct extensive experiments on four public databases to evaluate the proposed model under four deployment modes. Our findings demonstrate that PalmCohashNet outperforms competing methods in terms of authentication accuracy. 

# Training

```
python main.py  -b 256 --ds "casiam" --bits 32 --model res18 --comment caRes32 --epoch 4000

python main.py  -b 256 --ds "polyu" --bits 32 --model res18 --comment poRes32 --epoch 4000

python main.py  -b 256 --ds "tjppv" --bits 32 --model res18 --comment tjRes32 --epoch 4000

```


