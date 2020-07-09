![Microsoft Cloud Workshops](https://github.com/Microsoft/MCW-Template-Cloud-Workshop/raw/master/Media/ms-cloud-workshop.png "Microsoft Cloud Workshops")

<div class="MCWHeader1">
MLOps
</div>

<div class="MCWHeader2">
Hands-on lab step-by-step
</div>

<div class="MCWHeader3">
June 2020
</div>

Information in this document, including URL and other Internet Web site references, is subject to change without notice. Unless otherwise noted, the example companies, organizations, products, domain names, e-mail addresses, logos, people, places, and events depicted herein are fictitious, and no association with any real company, organization, product, domain name, e-mail address, logo, person, place or event is intended or should be inferred. Complying with all applicable copyright laws is the responsibility of the user. Without limiting the rights under copyright, no part of this document may be reproduced, stored in or introduced into a retrieval system, or transmitted in any form or by any means (electronic, mechanical, photocopying, recording, or otherwise), or for any purpose, without the express written permission of Microsoft Corporation.

Microsoft may have patents, patent applications, trademarks, copyrights, or other intellectual property rights covering subject matter in this document. Except as expressly provided in any written license agreement from Microsoft, the furnishing of this document does not give you any license to these patents, trademarks, copyrights, or other intellectual property.

The names of manufacturers, products, or URLs are provided for informational purposes only and Microsoft makes no representations and warranties, either expressed, implied, or statutory, regarding these manufacturers or the use of the products with any Microsoft technologies. The inclusion of a manufacturer or product does not imply endorsement of Microsoft of the manufacturer or product. Links may be provided to third party sites. Such sites are not under the control of Microsoft and Microsoft is not responsible for the contents of any linked site or any link contained in a linked site, or any changes or updates to such sites. Microsoft is not responsible for webcasting or any other form of transmission received from any linked site. Microsoft is providing these links to you only as a convenience, and the inclusion of any link does not imply endorsement of Microsoft of the site or the products contained therein.

© 2020 Microsoft Corporation. All rights reserved.

Microsoft and the trademarks listed at <https://www.microsoft.com/en-us/legal/intellectualproperty/Trademarks/Usage/General.aspx> are trademarks of the Microsoft group of companies. All other trademarks are property of their respective owners.

**Contents**

<!-- TOC -->

- [MLOps hands-on lab step-by-step](#mlops-hands-on-lab-step-by-step)
  - [Abstract and learning objectives](#abstract-and-learning-objectives)
  - [Overview](#overview)
  - [Solution architecture](#solution-architecture)
  - [Requirements](#requirements)
  - [Before the hands-on lab](#before-the-hands-on-lab)
  - [Exercise 1: Creating and evaluating compliance classification models](#exercise-1-creating-and-evaluating-compliance-classification-models)
    - [Task 1: Setup the notebooks environment](#task-1-setup-the-notebooks-environment)
    - [Task 2: Create the classification model using a notebook](#task-2-create-the-classification-model-using-a-notebook)
  - [Exercise 2: Registering the model](#exercise-2-registering-the-model)
    - [Task 1: Register Model using Azure Machine Learning Python SDK](#task-1-register-model-using-azure-machine-learning-python-sdk)
    - [Task 2: Register Model from Azure Machine Learning studio](#task-2-register-model-from-azure-machine-learning-studio)
  - [Exercise 3: Setup New Project in Azure DevOps](#exercise-3-setup-new-project-in-azure-devops)
    - [Task 1: Create New Project](#task-1-create-new-project)
    - [Task 2: Import Quickstart code from a GitHub Repo](#task-2-import-quickstart-code-from-a-github-repo)
    - [Task 3: Update the build YAML file](#task-3-update-the-build-yaml-file)
    - [Task 4: Create new Service Connection](#task-4-create-new-service-connection)
  - [Exercise 4: Setup and Run the Build Pipeline](#exercise-4-setup-and-run-the-build-pipeline)
    - [Task 1: Setup Build Pipeline](#task-1-setup-build-pipeline)
    - [Task 2: Run the Build Pipeline](#task-2-run-the-build-pipeline)
    - [Task 3: Review Build Artifacts](#task-3-review-build-artifacts)
    - [Task 4: Review Build Outputs](#task-4-review-build-outputs)
  - [Exercise 5: Setup the Release Pipeline](#exercise-5-setup-the-release-pipeline)
    - [Task 1: Create an Empty Job](#task-1-create-an-empty-job)
    - [Task 2: Add Build Artifact](#task-2-add-build-artifact)
    - [Task 3: Add Variables to Deploy and Test stage](#task-3-add-variables-to-deploy-and-test-stage)
    - [Task 4: Setup Agent Pool for Deploy and Test stage](#task-4-setup-agent-pool-for-deploy-and-test-stage)
    - [Task 5: Add Use Python Version task](#task-5-add-use-python-version-task)
    - [Task 6: Add Install Requirements task](#task-6-add-install-requirements-task)
    - [Task 7: Add Deploy and Test Webservice task](#task-7-add-deploy-and-test-webservice-task)
    - [Task 8: Define Deployment Trigger](#task-8-define-deployment-trigger)
    - [Task 9: Enable Continuous Deployment Trigger](#task-9-enable-continuous-deployment-trigger)
    - [Task 10: Save the Release Pipeline](#task-10-save-the-release-pipeline)
  - [Exercise 6: Test Build and Release Pipelines](#exercise-6-test-build-and-release-pipelines)
    - [Task 1: Make Edits to Source Code](#task-1-make-edits-to-source-code)
    - [Task 2: Monitor Build Pipeline](#task-2-monitor-build-pipeline)
    - [Task 3: Monitor Release Pipeline](#task-3-monitor-release-pipeline)
    - [Task 4: Review Release Pipeline Outputs](#task-4-review-release-pipeline-outputs)
  - [Exercise 7: Testing the deployed solution](#exercise-7-testing-the-deployed-solution)
    - [Task 1: Test the Deployment](#task-1-test-the-deployment)
  - [Exercise 8: Examining deployed model performance](#exercise-8-examining-deployed-model-performance)
    - [Task 1: Activate App Insights and data collection on the deployed model](#task-1-activate-app-insights-and-data-collection-on-the-deployed-model)
    - [Task 2: Check Application Insights telemetry](#task-2-check-application-insights-telemetry)
    - [Task 3: Check the data collected](#task-3-check-the-data-collected)
  - [After the hands-on lab](#after-the-hands-on-lab)
    - [Task 1: Clean up lab resources](#task-1-clean-up-lab-resources)

<!-- /TOC -->

# MLOps hands-on lab step-by-step

## Abstract and learning objectives

In this hands-on lab, you will learn how Trey Research can leverage Deep Learning technologies to scan through their vehicle specification documents to find compliance issues with new regulations. You will standardize the model format to ONNX and observe how this simplifies inference runtime code, enabling pluggability of different models and targeting a broad range of runtime environments, and most importantly, improves inferencing speed over the native model. You will build a DevOps pipeline to coordinate retrieving the latest best model from the model registry, packaging the web application, deploying the web application, and inferencing web service. After a first successful deployment, you will make updates to both the model, the and web application, and execute the pipeline once to achieve an updated deployment. You will also learn how to monitor the model's performance after it is deployed, so Trey Research can be proactive with performance issues.

At the end of this hands-on lab, you will be better able to implement end-to-end solutions that fully operationalize deep learning models, inclusive of all application components that depend on the model.

## Overview

Trey Research Inc. delivers innovative solutions for manufacturers. They specialize in identifying and solving problems for manufacturers that can run the range from automating away mundane but time-intensive processes to delivering cutting edge approaches that provide new opportunities for their manufacturing clients. Trey Research has decades specializing in data science and application development that until now were separate units. They have seen the value created by the ad-hoc synergies between data science and app development, but they would like to unlock the greater, long term value as they formalize their approach by combining the two units into one, and follow one standardized process for operationalizing their innovations.

As their first effort of this combined initiative, they would like to define a process for operationalizing deep learning that encompasses all phases of the application life cycle along with model creation and deployment of a deep learning model. For this first proof of concept (PoC), they would like to focus on component compliance. Specifically, they are looking to leverage Deep Learning technologies with Natural Language Processing (NLP) techniques to scan through vehicle specification documents to find compliance issues with new regulations. Even though this first scenario is focused on vehicle components, they believe this approach will generalize to any scenario involving an inventory of components, which all of their manufacturing customers deal with. The component descriptions, which are free form text, are entered and managed via a web application. This web application will take new component descriptions entered by authorized technicians and label the component as compliant or non-compliant, based on the text.

They want to ensure the overall process they create enables them to update both the underlying model and the web app in one, unified pipeline. They also want to be able to monitor the model's performance after it is deployed so they can be proactive with performance issues.

## Solution architecture

![The lab solution architecture as described by the text that follows.](media/architecture-overview.png 'Solution Architecture')

The overall approach used in this lab is to orchestrate continuous integration and continuous delivery Azure Pipelines from Azure DevOps. These pipelines are triggered by changes to artifacts that describe a machine learning pipeline, that is created with the Azure Machine Learning SDK. In the lab, you make a change to the model training script that executes the Azure Pipelines Build Pipeline, which trains the model and creates the container image. Then this triggers an Azure Pipelines Release pipeline that deploys the model as a web service to AKS, by using the Docker image that was created in the Build pipeline. Once in production, the scoring web service is monitored using a combination of Application Insights and Azure Storage.

## Requirements

1. Microsoft Azure subscription must be pay-as-you-go or MSDN.

   - Trial subscriptions will not work. You will run into issues with Azure resource quota limits.

   - Subscriptions with access limited to a single resource group will not work. You will need the ability to deploy multiple resource groups.

2. An Azure DevOps account.

## Before the hands-on lab

Refer to the [Before the hands-on lab setup guide](./Before&#32;the&#32;HOL&#32;-&#32;MLOps.md) before continuing to the lab exercises.

## Exercise 1: Creating and evaluating compliance classification models

Duration: 40 minutes

In this exercise, you create a model for classifying component text as compliant or non-compliant. This tutorial uses the cloud notebook server in your Azure Machine Learning workspace for an install-free and pre-configured experience, available in Azure Machine Learning studio.

> **Note:** The new [Azure Machine Learning studio](https://ml.azure.com) provides a new immersive experience for managing the end-to-end machine learning lifecycle. You can use it either by logging in directly to it or by selecting the **Try the new Azure Machine Learning studio, Launch now** option in the **Overview** section of your Azure Machine Learning workspace.

### Task 1: Setup the notebooks environment

1. Download the [**Deep Learning with Text.ipynb**](./notebooks/Deep&#32;Learning&#32;with&#32;Text.ipynb) notebook to your computer, by selecting the **Raw** view in GitHub, and then **right-click + Save as**. Please ensure that the extension of the saved file is `.ipynb`. This is the notebook you will step through executing in this lab.

2. Sign in to [Azure Machine Learning studio](https://ml.azure.com).

3. Select your subscription and the workspace you have available.

4. Select **Notebooks** on the left navigation pane.

    ![In Azure Machine Learning Studio, Notebooks is selected from the left navigation pane.](media/notebook-00.png 'Open notebooks in Azure Machine Learning Studio')

5. Select the folder under the **User files** section. It should be named as the currently logged user name. Select the option to **Create new folder** in the top menu.

    ![On the Notebooks screen, the current user is selected beneath the User Files section, and the Create New Folder icon is highlighted in the top toolbar.](media/notebook-01.png 'Create new notebooks folder')

6. Fill in the folder name: `MCW-MLOps`.

7. Select the **Upload files** option in the top menu.

    ![On the Notebooks screen, beneath user files, the folder of the current user is expanded displaying the folder that was created in the previous step. The Upload files icon is highlighted in the top toolbar.](media/notebook-02.png 'Upload notebook to the workspace file share')

8. Browse for the downloaded notebook, **Deep Learning with Text.ipynb** and then select **MCW-MLOps** folder as the target folder. Select **Upload**.

9. Select the notebook. Select **+ New Compute** to create the compute instance VM.

    ![On the Notebooks screen, beneath user files, the folder of the current user is expanded along with the folder that was created in step 6. Inside this folder the Notebook that we uploaded in the previous step is displayed and is selected. On the Compute screen to the right, the + New Compute button is highlighted in the top taskbar.](media/notebook-03.png 'Create new compute instance')

10. Provide the necessary data for creating a new compute instance to run on your notebooks.

    a. Compute name: `mlops-compute`. When you create a VM, provide a name. The name must be between 2 to 16 characters. Valid characters are letters, digits, and the - character, and must also be unique across your Azure subscription.

    b. Virtual machine type: CPU (Central Processing Unit)

    c. Virtual machine size: **Standard_D3_v2**.

    d. Then select **Create**. It can take approximately 5 minutes to set up your VM.

    ![The New Compute Instance form is displayed populated with the preceding values.](media/notebook-04.png 'Configure the new compute instance')

    > **Note**: Once the VM is available it will be displayed in the top toolbar.

11. Having selected the uploaded **Deep Learning with Text.ipynb** notebook, select the **Edit** drop down, and then select **Edit in Jupyter**. The new browser window will be opened.

    ![On the Notebooks screen, the Deep Learning with Text Notebook is selected. On the Compute screen to the right, a drop down list shows the compute currently running for the notebook, and in the taskbar the Edit option is expanded with the options Edit in Jupyter and Edit in JupyterLab.](media/notebook-05.png 'Edit the notebook in Jupyter')

12. Select **Python 3.6 - Azure ML** if you are asked to select a Kernel.

    ![A Kernel not found dialog is displayed with Python 3.6 - Azure ML selected from a drop down list. A Set Kernel button is available to confirm the kernel selection.](media/notebook-06.png 'Select Kernel version')

### Task 2: Create the classification model using a notebook

1. Follow the instructions within the notebook to complete the lab.

2. Back to the [Azure Machine Learning Studio](https://ml.azure.com), in **Notebooks**, under the **MCW-MLOps** folder, navigate to the **model** folder and download the **model.onnx** file to your local disk. We will use the downloaded model file in the next exercise.

   > **Note**: If the downloaded file name is changed to **utf-8 model.onnx** or **notebooks_model_model.onnx**, then rename the file back to `model.onnx`.

   > **Note**: The **model.onnx** file is generated during the execution of the notebook at the previous step (step 1). When running the notebook, make sure the execution is successful, and the file is correctly created.

## Exercise 2: Registering the model

Duration: 15 minutes

In this exercise, you explore the approaches you can take to managing the model versions, their association with Experiment Runs, and how you can retrieve the models both programmatically and via the [Azure Machine Learning studio](https://ml.azure.com).

### Task 1: Register Model using Azure Machine Learning Python SDK

1. Download the [**Register Model.ipynb**](./notebooks/Register&#32;Model.ipynb) notebook to your computer, by selecting the **Raw** view in GitHub, and then **right-click + Save as**. Please ensure that the extension of the saved file is `.ipynb`. This is the notebook you will step through executing in this exercise.

2. In the Studio, navigate to **Notebooks**, and select **Upload files** option in the top menu.

3. Browse your local computer for the downloaded notebook, **Register Model.ipynb** and then select **MCW-MLOps** as the target folder. Select **Upload**.

4. Select the notebook `Register Model.ipynb`. On the top bar, select the **mlops-compute** compute instance to use to run the notebook. Select the **Edit** drop down, and then select **Edit in Jupyter**. The new browser window will be opened.

5. Follow the instructions within the notebook to complete the lab.

6. Navigate back to the [Azure Machine Learning studio](https://ml.azure.com) either directly or via the [Azure Portal](https://portal.azure.com). Make sure you select the Azure Machine Learning workspace that you created from the notebook. Open your **Models** section, and observe the **version 1** of the registered model: `compliance-classifier`.

    ![In Azure Machine Learning Studio, from the left menu, Models is selected. In the Model List, the compliance-classifier with the version of 1 is highlighted.](media/model-registry-01.png 'Registered Model: compliance-classifier')

### Task 2: Register Model from Azure Machine Learning studio

1. In  [Azure Machine Learning studio](https://ml.azure.com), open your **Models** section and select **+ Register model**.

    ![In Auzre Machine Learning Studio, Models is selected in the left menu. In the taskbar of the Model list, the + Register Model button is selected.](media/model-registry-02.png 'Register Model in Azure Machine Learning studio')
  
2. Provide the following input to the **Register a model** dialog, and then select **Register**.

    a. Name: `compliance-classifier`

    b. Description: `Deep learning model to classify the descriptions of car components as compliant or non-compliant.`

    c. Model Framework: **TensorFlow**

    d. Model Framework Version: `2.0`

    e. Model file: Select the `model.onnx` file from your local disk.

    ![The Register a Model form is displayed populated with the preceding values.](media/model-registry-03.png 'Register a model Dialog')

3. Navigate to your **Models** section, and observe the **version 2** of the registered model: **compliance-classifier**.

    ![The Model list is displayed showing two rows containing both versions of the compliance-classifier model. Version 2 of the compliance-classifier model is highlighted in the list.](media/model-registry-04.png 'Registered Model: compliance-classifier version 2')

## Exercise 3: Setup New Project in Azure DevOps

Duration: 20 minutes

### Task 1: Create New Project

1. Sign in to [Azure DevOps](http://dev.azure.com).

2. Select **+ New project**.

    ![In the Azure DevOps screen, the + New project button is selected.](media/devops-project-01.png 'Create new project')

3. Provide Project Name: `mlops-quickstart` and select **Create**.

    ![The Create new project dialog is shown populated with the value above. Visibility is set to Private, and the Create button is highlighted.](media/devops-project-02.png 'Create New Project Dialog')

### Task 2: Import Quickstart code from a GitHub Repo

1. Within the new project:

   a. Select **Repos** from left navigation bar.

   b. Select **Import** from the content section.

    ![In Azure DevOps, Repos is selected from the left menu. In the mlops-quickstart screen the Import button is selected in the Import a repository section.](media/devops-project-03.png 'Azure DevOps Repos')

2. Provide the following GitHub URL: `https://github.com/solliancenet/mcw-mlops-starter-v2` and select **Import**. This should import the code required for the quickstart.

    ![In the Import a Git repository dialog, the Clone URL is populated with the value indicated above and the Import button is selected.](media/devops-project-04.png 'Import a Git repository dialog')

    > **Note**: If you receive an error while importing the repository, please disable the preview feature **New Repos landing pages** and import the GitHub repository from the old UI, as shown in steps #3, #4, and #5 below.

3. [Optional] Select **Account settings, Preview features**.

    ![On the mlops-quickstart repo page, Settings is expanded in the taskbar and the Preview features item is highlighted.](media/preview_features-01.png 'Preview features')

4. [Optional] From the list of preview features, disable the preview feature **New Repos landing pages**.

   ![A list of preview features is displayed highlighting the New Repos landing pages features.](media/preview_features-02.png 'Disable New Repos landing pages')

5. [Optional] Repeat Step #1 above to import the GitHub repository from the old UI.

### Task 3: Update the build YAML file

1. Select and open the **azure-pipelines.yml** file.

2. Select **Edit** and update the following variables: **resourcegroup**, and **workspace**. If you are using your own Azure subscription, please provide names to use. If an environment is provided to you, be sure to replace XXXXX in the values below with your unique identifier.

    ![In the left menu, beneath Repos, the Files item is selected. In the files list, azure-pipelines.yml is selected. In the editor pane, the contents of the file are displayed with the resource group and workspace variables highlighted. The Edit button is selected in the top toolbar.](media/devops-build-pipeline-01.png 'Edit Build YAML file')

3. Select **Commit** to save your changes and select **Commit** again in the commit properties dialog.

    ![The content of azure-pipelines.yml is updated and the Commit button is selected in the top taskbar.](media/devops-build-pipeline-02.png 'Commit Build YAML file')

### Task 4: Create new Service Connection

1. From the left navigation, select **Project settings** and then select **Service connections**.

    ![In the left menu, Project settings is selected. In the Project Settings list, Service connections is selected. In the Create your first service connection screen, the Create service connection button is selected.](media/devops-build-pipeline-03.png 'Service Connections')

2. Select **Create service connection**, select **Azure Resource Manager**, and then select **Next**.

    ![In the New service connection dialog, Azure Resource Manager is selected from the list. The Next button is selected.](media/devops-build-pipeline-04.png 'Azure Resource Manager')

3. Select **Service principal (automatic)** and then select **Next**.

    ![In the New service connection dialog, under Authentication method, Service principal (automatic) is selected. The Next button is selected.](media/devops-build-pipeline-05.png 'Service principal authentication')

4. Provide the following information in the **New Azure service connection** dialog box and then select **Save**:

    a. **Scope level**: **Machine Learning Workspace**

    b. **Subscription**: Select the Azure subscription to use.

    > **Note**: It might take up to 30 seconds for the **Subscription** dropdown to be populated with available subscriptions, depending on the number of different subscriptions your account has access to.

    c. **Resource group**: This value should match the value you provided in the **azure-pipelines.yml** file.

    d. **Machine Learning Workspace**: This value should match the value you provided in the **azure-pipelines.yml** file.

    e. **Service connection name**: `quick-starts-sc`

    f. **Security**: Grant access permission to all pipelines is checked.

    ![The New Azure service connection form is populated with the values outlined above. The Save button is selected.](media/devops-build-pipeline-06.png 'Add an Azure Resource Manager service connection dialog')

    >**Note**: If you are unable to select your **Machine Learning Workspace**, do the following steps:

    a. Quit the `New Azure service connection` dialog.
    
    b. Refresh or reload the web browser.
    
    c. Repeat steps 1-3 above.
    
    d. In step 4, change the `Scope level` to **Subscription** and then select your **Resource group**.
    
    e. Please remember to name your service connection as `quick-starts-sc`.
    
    f. Grant access permission to all pipelines.

## Exercise 4: Setup and Run the Build Pipeline

Duration: 30 minutes

### Task 1: Setup Build Pipeline

1. From left navigation select **Pipelines, Pipelines** and then select **Create pipeline**.

    ![In the left menu, the Pipelines item is expanded with the Pipelines item selected. In the content pane, the Create pipeline button is highlighted.](media/devops-build-pipeline-07.png 'Create Build Pipeline')

2. Select **Azure Repos Git** as your code repository.

    ![On the Connect tab of your pipeline screen, Azure Repos Git is selected beneath the Where is your code? prompt.](media/devops-build-pipeline-08.png 'Repository Source')

3. Select **mlops-quickstart** as your repository.

    ![On the Select tab of your pipeline screen, beneath the Select a repository section, the mlops-quickstart repository is selected.](media/devops-build-pipeline-09.png 'Select Repository')

4. Review the YAML file.

    The build pipeline has four key steps:

    a. Attach folder to workspace and experiment. This command creates the **.azureml** subdirectory that contains a **config.json** file that is used to communicate with your Azure Machine Learning workspace. All subsequent steps rely on the **config.json** file to instantiate the workspace object.

    b. Create the AML Compute target to run your master pipeline for model training and model evaluation.

    c. Run the master pipeline. The master pipeline has two steps: (1) Train the machine learning model, and (2) Evaluate the trained machine learning model. The evaluation step evaluates if the new model performance is better than the currently deployed model. If the new model performance is improved, the evaluate step will create a new Image for deployment. The results of the evaluation step will be saved in a file called **eval_info.json** that will be made available for the release pipeline. You can review the code for the master pipeline and its steps in **aml_service/pipelines_master.py**, **scripts/train.py**, and **scripts/evaluate.py**.

    d. Publish the build artifacts. The **snapshot of the repository**, **config.json**, and **eval_info.json** files are published as build artifacts and thus can be made available for the release pipeline.

    ![On the Review tab of your pipeline screen, the contents of azure-pipelines.yml is displayed.](media/devops-build-pipeline-10.png 'Build pipeline YAML')

### Task 2: Run the Build Pipeline

1. Select **Run** to start running your build pipeline.

    ![On the Review tab of your pipeline screen, the contents of azure-pipelines.yml is displayed. The Run button is selected from the top taskbar.](media/devops-build-pipeline-11.png 'Run Build Pipeline')

2. Monitor the build run. The build pipeline, for the first run, will take around 25 minutes to run.

    ![A build pipeline run summary screen is displayed indicating it was manually triggered. A single job is selected in the Jobs section with a status of Success.](media/devops-build-pipeline-12.png 'Monitor Build Pipeline')

3. Select **Job** to monitor the detailed status of the build pipeline execution.

    ![In the Jobs screen, the Job that was selected in the previous step is expanded displaying multiple steps. To the right a summary of the run is displayed.](media/devops-build-pipeline-13.png 'Monitor Build Pipeline Details')

### Task 3: Review Build Artifacts

1. The build will publish an artifact named `devops-for-ai`. Select **1 published** to review the artifact contents.

    ![On the build pipeline run summary, in the table outlining the manual run, the 1 published beneath the related column is selected.](media/devops-build-pipeline-14.png 'Build Artifacts')

2. Select **outputs, eval_info.json**, and then select the download arrow. The `eval_info.json` is the output from the *model evaluation* step. The information from the evaluation step will be used in the release pipeline to deploy the model. Select the back arrow to return to the previous screen.

    ![On the Artifacts screen, a list of files are displayed. The outputs folder is expanded and the eval-info.json file download button is selected. The back arrow is selected a the top of the screen to navigate back to the previous page.](media/devops-build-pipeline-15.png 'Download JSON file')

3. Open the **eval_info.json** in a json viewer or a text editor and observe the information. The json output contains information such as if the model passed the evaluation step (**deploy_model**: **true** or **false**), and the name and id of the created image (**image_name** and **image_id**) to deploy.

    ![The contents of the eval_info.json file is displayed in Visual Studio.](media/devops-build-pipeline-16.png 'Eval Info JSON File')

### Task 4: Review Build Outputs

1. Log in to [Azure Machine Learning studio](https://ml.azure.com) either directly or via the [Azure Portal](https://portal.azure.com). Make sure you select the Azure Machine Learning workspace that you created from the notebook earlier. Open your **Models** section, and observe the versions of the registered model: **compliance-classifier**. The latest version is the one registered by the build pipeline you ran in the previous task.

    ![In Azure Machine Learning Studio, models is selected in the left menu. A list of multiple versions of the compliance-classifier model is displayed in the Model list table with the largest version of the model highlighted in the table.](media/devops-build-outputs-01.png 'Registered Models in Azure Machine Learning studio')

2. Select the latest version of your model to review its properties. The **build_number** tag links the registered model to the Azure DevOps build that generated it.

    ![The model screen for the latest version of the compliance-classifier is displayed with the Build_Number tag highlighted.](!media/../media/devops-build-outputs-02.png 'Registered model details and Build_Number tag')

3. Open your **Datasets** section and observe the versions of the registered dataset: **connected_car_components**. The latest version is the one registered by the build pipeline you have run in the previous task.

    ![In Azure Machine Learning Studio, Datasets is selected from the left menu. The Datasets screen displays a list of datasets on the Registered datasets tab. The connected_car_components dataset is selected in the table with its tag indicating the build number matching the value from the previous step.](media/devops-build-outputs-03.png 'Registered Datasets in Azure Machine Learning studio')

4. Select the latest version of your dataset to review its properties. Notice the **build_number** tag that links the dataset version to the Azure DevOps build that generated it.

    ![The registered dataset version properties screen is displayed with a tag that indicates the build_number.](medial/../media/devops-build-outputs-04.png 'Registered dataset details in Azure Machine Learning studio')

5. Select **Models** to view a list of registered models that reference the dataset.

    ![A list of registered models that reference the selected dataset is displayed.](media/devops-build-outputs-05.png 'Registered dataset model references in Azure Machine Learning studio')

6. Log in to the [Azure Portal](https://portal.azure.com), open your **Resource Group, Workspace, Images** section, and observe the deployment image created during the build pipeline: `compliance-classifier-image`.

    ![In the Images resource page, Images is selected from the left menu, and a list of Images is displayed. Multiple compliance-classifier-image images are shown with their associated versions. The latest version image is highlighted in the table.](media/devops-build-outputs-06.png 'Images in Azure Portal')

## Exercise 5: Setup the Release Pipeline

Duration: 20 minutes

### Task 1: Create an Empty Job

1. Return to Azure DevOps and navigate to **Pipelines, Releases** and select **New pipeline**.

    ![In Azure DevOps, the Pipelines item is expanded in the left menu with the Releases item selected. In the content pane, a message indicates No release pipelines found and the New pipeline button is selected.](media/devops-release-pipeline-01.png 'New Release Pipeline')

2. Select **Empty job**.

    ![In the Select a template dialog, the Empty job item is selected above the list featured templates.](media/devops-release-pipeline-02.png 'Select a template: Empty job')

3. Provide Stage name: `Deploy and Test` and close the dialog.

    ![On the Stage dialog, the Stage name textbox is populated with Deploy and Test. The close button at the top of the dialog is selected.](media/devops-release-pipeline-03.png 'Deploy and Test Stage')

### Task 2: Add Build Artifact

1. Select **+ Add an artifact**.

    ![In the New release pipeline screen, on the Pipeline tab, the + Add a new artifact item is selected within the Artifacts tile.](media/devops-release-pipeline-04.png 'Add an artifact')

2. Select **Source type**: **Build**, **Source (build pipeline)**: **mlops-quickstart**.

    > **Note**: Observe the note that shows that the mlops-quickstart publishes the build artifact named devops-for-ai.

    Finally, select **Add**.

    ![The Add an artifact dialog form is populated with the aforementioned values. The Add button is selected.](media/devops-release-pipeline-05.png 'Add a build artifact')

### Task 3: Add Variables to Deploy and Test stage

1. Open **View stage tasks** link.

    ![On the New release pipeline screen, within the Deploy and Test tile, the View stage tasks link is selected.](media/devops-release-pipeline-06.png 'View Stage Tasks')

2. Open the **Variables** tab.

    ![On the New release pipeline screen, the Variables tab is selected.](media/devops-release-pipeline-07.png 'Release Pipeline Variables')

3. Add four Pipeline variables as name - value pairs and then select **Save** (use the default values in the **Save** dialog):

    a. **Name**: `aks_name` **Value**: `aks-cluster01`

    b. **Name**: `aks_region` **Value**: `eastus`

    c. **Name**: `service_name` **Value**: `compliance-classifier-service`

    d. **Name**: `description` **Value**: `"Compliance Classifier Web Service"`

    > **Note**: Include the double quotes around the **description** value!
    

    > **Note**:
    >   - Keep the scope for the variables to the **Deploy and Test** stage.
    >   - The name of the Azure region should be the same one that was used to create Azure Machine Learning workspace earlier on.

      ![On the New release pipeline screen, the Variables tab is selected, and the Pipeline variables item is selected from a list of variable types. The list of Pipeline variables is displayed showing the variables that were just created.](media/devops-release-pipeline-08.png 'Add Pipeline Variables')

### Task 4: Setup Agent Pool for Deploy and Test stage

1. Open the **Tasks** tab.

    ![On the New release pipeline screen, the Tasks tab is selected.](media/devops-release-pipeline-09.png 'Pipeline Tasks')

2. Select **Agent job** and set **Agent pool** to `Azure Pipelines` and change **Agent Specification** to `ubuntu-16.04`.

    ![On the New release pipeline screen, Tasks tab, the Agent job is selected. The Agent job details form is populated with the aforementioned values.](media/devops-release-pipeline-10.png 'Agent Job Setup')

### Task 5: Add Use Python Version task

1. Select **Add a task to Agent job** (the **+** button), search for `Use Python Version`, and select **Add**.

    ![On the New release pipeline screen, the Tasks tab is selected. Agent job is displayed in the list of tasks. The + button is selected in the Agent job tile. The Add tasks pane is displayed with Use Python version entered in the search box, and the Use Python version search result is highlighted with its Add button selected.](media/devops-release-pipeline-11.png 'Add Use Python Version Task')

2. Provide **Display name:** `Use Python 3.6` and **Version spec:** `3.6`

    ![The Use Python version form is displayed populated with the preceding values.](media/devops-release-pipeline-12.png 'Use Python Version Task Dialog')

### Task 6: Add Install Requirements task

1. Select **Add a task to Agent job** (the **+** button), search for `Bash`, and select **Add**.

    ![On the New release pipeline screen, the Tasks tab is selected. A list of tasks includes the Agent job that has a Use Python 3.6 task. The + button is selected in the Agent job tile, and in the Add tasks form, Bash is entered into the search text box and the Bash item is highlighted in a list of search results with its Add button selected.](media/devops-release-pipeline-13.png 'Add Bash Task')

2. Provide **Display name:** `Install Requirements` and select **Browse script path ...** to provide **Script Path**.

    ![On the New release pipeline screen, the Tasks tab is selected. A list of tasks is displayed below the Agent job. The Install requirements job is selected from this list. In the Bash form, fields are populated with the preceding values.](media/devops-release-pipeline-14.png 'Bash Task Dialog')

3. Navigate to **Linked artifacts/_mlops-quickstart (Build)/devops-for-ai/environment_setup** and select **install_requirements.sh**.

    ![A Select a file or folder dialog is displayed showing the location hierarchy to the install_requirements.sh file. The OK button is selected.](media/devops-release-pipeline-15.png 'Select Path Dialog')

4. Expand **Advanced** and select **Browse working directory ...** to provide **Working Directory**.

    ![On the Bash form, the Advanced section is expanded. The ellipsis button is selected next to the Working Directory textbox.](media/devops-release-pipeline-16.png 'Bash Task - Advanced Section')

5. Navigate to **Linked artifacts/_mlops-quickstart (Build)/devops-for-ai** and select **environment_setup**.

    ![In the Select a file or folder dialog, the location hierarchy is displayed to the environment_setup folder.](media/devops-release-pipeline-17.png 'Select Path Dialog')

### Task 7: Add Deploy and Test Webservice task

1. Select **Add a task to Agent job** (the **+** button), search for `Azure CLI`, and select **Add**.

    ![In the New release pipeline screen, the Tasks tab is selected. In the list of Tasks, in the Agent job tile, the + button is selected. In the Add tasks form, Azure CLI is entered into the search box, and the Azure CLI search result is highlighted with its Add button selected.](media/devops-release-pipeline-18.png 'Azure CLI Task')

2. Provide the following information for the Azure CLI task:

    a. **Task version**: `1.*`

    b. **Display name**: `Deploy and Test Webservice`

    c. **Azure subscription**: `quick-starts-sc`

    > **Note**: This is the service connection we created in Exercise 1 / Task 4.

    d. **Script Location**: `Inline script`

    e. **Inline Script**: `python aml_service/deploy.py --service_name $(service_name) --aks_name $(aks_name) --aks_region $(aks_region) --description $(description)`

    ![On the Tasks tab of the New release pipeline screen, the Deploy and Test Webservice task is selected beneath the Agent job item. The Azure CLI form is populated with the preceding values.](media/devops-release-pipeline-19.png 'Azure CLI Task Dialog')

3. Expand **Advanced** and provide **Working Directory:** `$(System.DefaultWorkingDirectory)/_mlops-quickstart/devops-for-ai`.

    ![In the Azure CLI form, the Advanced section is expanded and the Working Directory field is populated with the specified value.](media/devops-release-pipeline-20.png 'Azure CLI Task - Working Directory')

> **Note**: Please review the code in `aml_service/deploy.py`. By using the `eval_info.json` file that is associated with each trained model, this script can determine if the new trained model's performance is better than the currently deployed model. If it determines the new model has better performance, it will deploy the new model to production in an **Azure Kubernetes Service (AKS)** cluster.

### Task 8: Define Deployment Trigger

1. Navigate to **Pipeline** tab, and select **Pre-deployment conditions** for the **Deploy and Test** stage.

2. Select **After release**.

    ![In the New release pipeline screen, the Pipelines tab is selected. Within the Stages tile, the Pre-deployment Condition item is selected in the Deploy and Test stage tile. The Pre-deployment conditions form is displayed with the Triggers section expanded. After release is selected as the trigger.](media/devops-release-pipeline-21.png 'Pre-deployment Conditions Dialog')

3. Close the dialog.

### Task 9: Enable Continuous Deployment Trigger

1. Select **Continuous deployment trigger** for `_mlops-quickstart` artifact.

2. Enable: **Creates a release every time a new build is available**.

    ![In the New release pipeline screen, the Pipelines tab is selected. Within the Artifacts tile, the Continuous deployment trigger option is selected on the _mlops-quickstart tile. The Continuous deployment trigger form is displayed indicating that it is Enabled.](media/devops-release-pipeline-22.png 'Continuous Deployment Trigger Dialog')

3. Close the dialog.

### Task 10: Save the Release Pipeline

1. Provide name: `mlops-quickstart-release`.

2. Select **Save** (use the default values in the **Save** dialog).

    ![In the header of the New pipeline screen, the pipeline name is set to mlops-quickstart-release, and the Save button is selected from the top taskbar.](media/devops-release-pipeline-23.png 'Save')

## Exercise 6: Test Build and Release Pipelines

Duration: 30 minutes

### Task 1: Make Edits to Source Code

1. Navigate to: **Repos -> Files -> scripts -> `train.py`**.

2. **Edit** `train.py`.

3. Change the **learning rate (lr)** for the optimizer from **0.1** to **0.001**.

4. Change the number of training **epochs** from **3** to **5**.

5. Select **Commit**.

    ![In Azure DevOps, the Repos item is expanded in the left menu with Files selected. In the file list, beneath the expanded scripts folder, train.py is selected. The train.py contents is displayed with the source code changes indicated in the previous steps in place. The Commit button is selected in the top toolbar.](media/devops-test-pipelines-01.png 'Edit Train.py')

6. Provide comment: `Improving model performance: changed learning rate.` and select **Commit**.

    ![On the Commit dialog, the comment above is entered and the Commit button is selected.](media/devops-test-pipelines-02.png 'Commit - Comment')

### Task 2: Monitor Build Pipeline

1. Navigate to **Pipelines, Pipelines**. Observe that the CI build is triggered because of the source code change.

   ![In Azure DevOps, Pipelines is selected below the Pipeline item in the left menu. In the Pipelines list, the Recent tab is selected. A currently running pipeline is shown and selected in this list.](media/devops-test-pipelines-03.png 'Pipelines - pipelines')

2. Select the pipeline run and monitor the pipeline steps. The pipeline will run for 18-25 minutes. Proceed to the next task when the build pipeline successfully completes.

   ![In the Job details screen, the progress of the pipeline run is shown. Every step shows success.](media/devops-test-pipelines-04.png 'Build Pipeline Steps')

### Task 3: Monitor Release Pipeline

1. Navigate to **Pipelines, Releases**. Observe that the Release pipeline is automatically triggered upon successful completion of the build pipeline. Select as shown in the figure to view pipeline logs.

   ![In Azure DevOps, on the left menu, Pipelines is expanded and the Releases item is selected. The ml-ops-quickstart-release screen is displayed with the Releases tab selected. A release named Release-1 is highlighted with a button in the Stage column that is used to view the pipeline logs.](media/devops-test-pipelines-05.png 'Pipelines - Releases')

2. The release pipeline will run for about 15 minutes. Proceed to the next task when the release pipeline successfully completes.

### Task 4: Review Release Pipeline Outputs

1. From the pipeline logs view, select **Deploy and Test Webservice** task to view details.

    ![In the Release-1, Deploy and Test Stage results, A list of tasks related to Agent job is displayed. The Deploy and Test Webservice task is selected from the list.](media/devops-test-pipelines-06.png 'Pipeline Logs')

2. Observe the **Scoring URI** and **API Key** for the deployed webservice. Please note down both the **Scoring URI** and **API Key** for **Exercise 8**.

    ![The Deploy and Test Webservice task logs are displayed. Within the logs the Webservice URI and Webservice API Key are highlighted.](media/devops-test-pipelines-07.png 'Deploy and Test Webservice Task Logs')

3. Log in to Azure Machine Learning studio. Open your **Endpoints** section, and observe the deployed webservice: **compliance-classifier-service**.

    ![In Azure Machine Learning Studio, the Endpoints item is selected from the left menu. In the list of Endpoints, the compliance-classifier-service is selected.](media/devops-test-pipelines-08.png 'Azure Machine Learning studio - Workspace, Deployments')

## Exercise 7: Testing the deployed solution

Duration: 15 minutes

In this exercise, you verify that the first release of the application works.

### Task 1: Test the Deployment

1. Download the [**Test Deployment.ipynb**](./notebooks/Test&#32;Deployment.ipynb) notebook to your computer, by selecting the **Raw** view in GitHub, and then **right-click + Save as**. Please ensure that the extension of the saved file is `.ipynb`. This is the notebook you will step through executing in this exercise.

2. In the Azure Machine Learning Studio, navigate to **Notebooks**, and select **Upload files** option in the top menu.

3. Browse your local computer for the downloaded notebook, **Test Deployment.ipynb** and then select **MCW-MLOps** folder as the target folder. Select **Upload**.

4. On the top bar, select the **mlops-compute** compute instance to use to run the notebook. Select the **Edit** drop down, and then select **Edit in Jupyter**. The new browser window will be opened.

5. Note that you will have to provide values for **Scoring URI** and **API Key** for the deployed webservice in the notebook.

6. Follow the instructions within the notebook to complete the task.

## Exercise 8: Examining deployed model performance

Duration: 15 minutes

In this exercise, you learn how to monitor the performance of a deployed model.

### Task 1: Activate App Insights and data collection on the deployed model

1. Download the [**Model Telemetry.ipynb**](./notebooks/Model&#32;Telemetry.ipynb) notebook to your computer, by selecting the **Raw** view in GitHub, and then **right-click + Save as**. Please ensure that the extension of the saved file is `.ipynb`. This is the notebook you will step through executing in this exercise.

2. In the Azure Machine Learning Studio, navigate to **Notebooks**, and select **Upload files** option in the top menu.

3. Browse your local computer for the downloaded notebook, **Model Telemetry.ipynb** and then select **MCW-MLOps** folder as the target folder. Select **Upload**.

4. On the top bar, select the **mlops-compute** compute instance to use to run the notebook. Select the **Edit** drop down, and then select **Edit in Jupyter**. The new browser window will be opened.

5. Follow the instructions within the notebook to complete the task. When finished, your deployed model has now both [Application Insights](https://docs.microsoft.com/en-us/azure/azure-monitor/app/app-insights-overview) integration and data collection activated.

6. Note that if there are errors (for example, **Too many requests for service compliance-classifier-service (overloaded)** or **No ready replicas for service compliance-classifier-service.**) when you make calls against the deployed web service after your enable app insights (last cell in the **Model Telemetry** notebook), you should wait for 5 minutes and rerun the cell to make the calls.

### Task 2: Check Application Insights telemetry

1. Navigate to the [Azure portal](https://portal.azure.com) and locate the resource group you created for this lab (the one where the Azure Machine Learning service workspace was created in).

2. Locate the Application Insights instance in the resource group and select it.

    ![The Application Insights resource is selected in the resource group.](media/model-telemetry-01.png 'Resource Group Overview')

3. Go to **Overview**.

4. From the top row of the right section, select **Logs**. This will open the Application Insights query editor with an empty new query. Dismiss the **Example queries** popup if displayed.

    ![On the Application Insights resource screen, the Overview item is selected in the left menu, and the Logs item is selected from the top toolbar.](media/model-telemetry-02.png 'Application Insights - Dashboard')

5. In the left pane, make sure the **Tables** tab is selected.

6. Hover over **requests** and select the icon on the right side - "Show sample records from this table". Next, select **Run**.

    ![On the Application Insights Logs screen, a New Query tab is shown with the Tables tab selected. The icon next to the requests table is selected.](media/model-telemetry-03.png 'Create Requests Query')

7. Look at the results displayed. Application Insights is tracing all requests made to your model. Sometimes, a couple of minutes are needed for the telemetry information to propagate. If there are no results displayed, wait a minute. Call your model wait a minute and select **Run** to re-execute the Application Insights query.

   ![On the Application Insights Logs screen, a query is run against the requests table and a list of results is displayed.](media/model-telemetry-04.png 'Requests Query Results')

> **Note**: If you do not see telemetry information after selecting **Run** to re-execute the Application insights query. Please rerun the last cell in the **Model Telemetry** notebook a few more times to generate more data. Then select **Run** on this page to re-execute the Application insights query.

### Task 3: Check the data collected

1. Navigate to the [Azure portal](https://portal.azure.com) and locate the resource group you created for this lab (the one where the Azure Machine Learning service workspace was created in).

2. Locate the Storage Account instance in the resource group and select it.

    ![The Storage account resource is selected in the resource group.](media/model-telemetry-05.png 'Resource Group Overview')

3. Go to **Storage Explorer (preview)**.

4. Expand the **BLOB CONTAINERS** section and identify the **modeldata** container. Select **More->Refresh** if you do not see **modeldata** container.

    ![In the Storage Explorer screen, the BLOB CONTAINERS item is expanded with the modeldata item selected.](media/model-telemetry-06.png 'Storage Explorer')

5. Identify the CSV files containing the collected data. The path to the output blobs is based on the following structure:

    **modeldata -> subscriptionid -> resourcegroup -> workspace -> webservice -> model -> version -> identifier -> year -> month -> day -> inputs.csv**

    ![In the Storage Explorer, the modeldata container is selected beneath BLOB containers. The inputs.csv file is selected in the list of files at the path specified above.](media/model-telemetry-07.png 'Storage Explorer - inputs.csv')

## After the hands-on lab

Duration: 5 minutes

To avoid unexpected charges, it is recommended that you clean up all of your lab resources when you complete the lab.

### Task 1: Clean up lab resources

1. Navigate to the Azure Portal and locate the resource group you created for this lab (the one where the Azure Machine Learning service workspace was created in).

2. Select **Delete resource group** from the command bar.

    ![Screenshot of the Delete resource group button.](media/cleanup-01.png 'Delete resource group button')

3. In the confirmation dialog that appears, enter the name of the resource group and select **Delete**.

4. Wait for the confirmation that the Resource Group has been successfully deleted. If you don't wait, and the delete fails for some reason, you may be left with resources running that were not expected. You can monitor using the Notifications dialog, which is accessible from the Alarm icon.

    ![The Notifications dialog box has a message stating that the resource group is being deleted.](media/cleanup-02.png 'Delete Resource Group Notification Dialog')

5. When the Notification indicates success, the cleanup is complete.

    ![The Notifications dialog box has a message stating that the resource group has been deleted.](media/cleanup-03.png 'Delete Resource Group Notification Dialog')

You should follow all steps provided _after_ attending the Hands-on lab.