
1. Visit the repository at [McGill-NLP/llm2vec](https://github.com/McGill-NLP/llm2vec) to install the environment, and go to Hugging Face to apply for access to the Mistral 7B model (requires CUDA environment).
2. Prepare your XLSX file, such as `oral_all_demo.xlsx` used in this project.
3. Run the following command to generate the feature matrix:

    ```bash
    CUDA_VISIBLE_DEVICES=0 python encode_hksts.py
    ```

    (In this script, specify the path to your XLSX file. It will generate the corresponding feature matrix and save it as `hksts.npy` in the current directory.)

4. Run the clustering script:

    ```bash
    python cluster_try.py
    ```

    (In this script, specify the number of sessions `k`, and adjust the minimum and maximum number of papers per session. The output will be saved as `oral_all_demo_with_clusterid.xlsx`. Clusterid ranges from 0 to k-1)