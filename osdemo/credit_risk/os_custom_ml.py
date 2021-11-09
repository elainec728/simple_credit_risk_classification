#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ibm-wos-utils==2.1.1
Python 3.7
"""
import warnings

warnings.filterwarnings('ignore')
import random
import time
import uuid

import matplotlib.pyplot as plt
import pandas as pd
import requests
from ibm_cloud_sdk_core.authenticators import CloudPakForDataAuthenticator
from ibm_watson_openscale import APIClient
from ibm_watson_openscale.base_classes.watson_open_scale_v2 import *
from ibm_watson_openscale.supporting_classes.enums import *
from ibm_watson_openscale.supporting_classes.payload_record import PayloadRecord
from ibm_wos_utils.drift.drift_trainer import DriftTrainer


def get_scoring_payload(df, cols_to_remove, no_of_records_to_score=1):
    for col in cols_to_remove:
        if col in df.columns:
            del df[col]

    fields = df.columns.tolist()
    values = df[fields].values.tolist()

    payload_scoring = {"fields": fields, "values": values[:no_of_records_to_score]}
    return payload_scoring


def custom_ml_scoring(scoring_url, payload_scoring):
    header = {"Content-Type": "application/json", "x": "y"}
    scoring_response = requests.post(scoring_url, json=payload_scoring, headers=header, verify=False)
    jsonify_scoring_response = scoring_response.json()
    return jsonify_scoring_response


def payload_logging(payload_scoring, scoring_response, wos_client, payload_data_set_id):
    scoring_id = str(uuid.uuid4())
    records_list = []

    # manual PL logging for custom ml provider
    pl_record = PayloadRecord(scoring_id=scoring_id, request=payload_scoring, response=scoring_response,
                              response_time=int(460))
    records_list.append(pl_record)
    wos_client.data_sets.store_records(data_set_id=payload_data_set_id, request_body=records_list)

    time.sleep(5)
    pl_records_count = wos_client.data_sets.get_records_count(payload_data_set_id)
    print("Number of records in the payload logging table: {}".format(pl_records_count))
    return scoring_id


def auth_cpd(WOS_CREDENTIALS):
    authenticator = CloudPakForDataAuthenticator(
        url=WOS_CREDENTIALS['url'],
        username=WOS_CREDENTIALS['username'],
        password=WOS_CREDENTIALS['password'],
        disable_ssl_verification=True
    )

    wos_client = APIClient(service_url=WOS_CREDENTIALS['url'], authenticator=authenticator)
    print(wos_client.version)
    print(wos_client.data_marts.show())
    return wos_client


def remove_existing_service_provider(wos_client, SERVICE_PROVIDER_NAME):
    service_providers = wos_client.service_providers.list().result.service_providers
    for service_provider in service_providers:
        service_instance_name = service_provider.entity.name
        if service_instance_name == SERVICE_PROVIDER_NAME:
            service_provider_id = service_provider.metadata.id
            wos_client.service_providers.delete(service_provider_id)
            print("Deleted existing service_provider for WML instance: {}".format(service_provider_id))


def add_service_provider(SERVICE_PROVIDER_NAME, SERVICE_PROVIDER_DESCRIPTION, ):
    request_headers = {"Content-Type": "application/json", "Custom_header_X": "Custom_header_X_value_Y"}
    MLCredentials = {}
    added_service_provider_result = wos_client.service_providers.add(
        name=SERVICE_PROVIDER_NAME,
        description=SERVICE_PROVIDER_DESCRIPTION,
        service_type=ServiceTypes.CUSTOM_MACHINE_LEARNING,
        request_headers=request_headers,
        operational_space_id="production",
        credentials=MLCredentials,
        background_mode=False
    ).result
    service_provider_id = added_service_provider_result.metadata.id
    print(wos_client.service_providers.get(service_provider_id).result)
    print('Service Provider ID : ' + service_provider_id)
    return service_provider_id


def remove_existing_subscription(wos_client, SUBSCRIPTION_NAME):
    subscriptions = wos_client.subscriptions.list().result.subscriptions
    for subscription in subscriptions:
        if subscription.entity.asset.name == "[asset] " + SUBSCRIPTION_NAME:
            sub_model_id = subscription.metadata.id
            wos_client.subscriptions.delete(subscription.metadata.id)
            print('Deleted existing subscription for model', sub_model_id)


def create_monitor(data_mart_id, target, parameters, thresholds, type, wos_client):
    type_dict = {'fairness': wos_client.monitor_definitions.MONITORS.FAIRNESS.ID,
                 'quality': wos_client.monitor_definitions.MONITORS.QUALITY.ID,
                 'drift': wos_client.monitor_definitions.MONITORS.DRIFT.ID}

    monitor_details = wos_client.monitor_instances.create(
        data_mart_id=data_mart_id,
        background_mode=False,
        monitor_definition_id=type_dict[type],
        target=target,
        parameters=parameters,
        thresholds=thresholds).result

    monitor_instance_id = monitor_details.metadata.id
    return monitor_instance_id


def finish_explanation_tasks(wos_client, explanation_task_ids, sample_size):
    finished_explanations = []
    finished_explanation_task_ids = []

    # Check for the explanation task status for finished status.
    # If it is in-progress state, then sleep for some time and check again.
    # Perform the same for couple of times, so that all tasks get into finished state.
    for i in range(0, 5):
        # for each explanation
        print('iteration ' + str(i))

        # check status for all explanation tasks
        for explanation_task_id in explanation_task_ids:
            if explanation_task_id not in finished_explanation_task_ids:
                result = wos_client.monitor_instances.get_explanation_tasks(
                    explanation_task_id=explanation_task_id).result
                print(result)
                print(explanation_task_id + ' : ' + result.entity.status.state)
                if (
                        result.entity.status.state == 'finished' or result.entity.status.state == 'error') and explanation_task_id not in finished_explanation_task_ids:
                    finished_explanation_task_ids.append(explanation_task_id)
                    finished_explanations.append(result)

        # if there is altest one explanation task that is not yet completed, then sleep for sometime,
        # and check for all those tasks, for which explanation is not yet completeed.

        if len(finished_explanation_task_ids) != sample_size:
            print('sleeping for some time..')
            time.sleep(10)
        else:
            break

    return finished_explanations


def construct_explanation_features_map(feature_name, feature_weight, explanation_features_map):
    if feature_name in explanation_features_map:
        explanation_features_map[feature_name].append(feature_weight)
    else:
        explanation_features_map[feature_name] = [feature_weight]


def score(training_data_frame):
    # The data type of the label column and prediction column should be same .
    # User needs to make sure that label column and prediction column array should have the same unique class labels
    prediction_column_name = "prediction"
    probability_column_name = "probability"

    feature_columns = list(training_data_frame.columns)
    training_data_rows = training_data_frame[feature_columns].values.tolist()

    payload_scoring_records = {
        "fields": feature_columns,
        "values": [x for x in training_data_rows]
    }

    header = {"Content-Type": "application/json", "x": "y"}
    scoring_response_raw = requests.post(scoring_url, json=payload_scoring_records, headers=header, verify=False)
    scoring_response = scoring_response_raw.json()

    prob_col_index = list(scoring_response.get('fields')).index(probability_column_name)
    predict_col_index = list(scoring_response.get('fields')).index(prediction_column_name)

    if prob_col_index < 0 or predict_col_index < 0:
        raise Exception("Missing prediction/probability column in the scoring response")

    import numpy as np
    probability_array = np.array([value[prob_col_index] for value in scoring_response.get('values')])
    prediction_vector = np.array([value[predict_col_index] for value in scoring_response.get('values')])

    return probability_array, prediction_vector


def generating_drift_model(training_df, drift_detection_input, scoring_method):
    drift_trainer = DriftTrainer(training_df, drift_detection_input)
    if model_type != "regression":
        drift_trainer.generate_drift_detection_model(scoring_method, batch_size=training_df.shape[0])
    # Note: Two column constraints are not computed beyond two_column_learner_limit(default set to 200)
    drift_trainer.learn_constraints(two_column_learner_limit=200)
    drift_trainer.create_archive()


def remove_drift_monitor_for_subscription(wos_client, subscription_id):
    monitor_instances = wos_client.monitor_instances.list().result.monitor_instances
    for monitor_instance in monitor_instances:
        monitor_def_id = monitor_instance.entity.monitor_definition_id
        if monitor_def_id == "drift" and monitor_instance.entity.target.target_id == subscription_id:
            wos_client.monitor_instances.delete(monitor_instance.metadata.id)
            print('Deleted existing drift monitor instance with id: ', monitor_instance.metadata.id)


def fairness(wos_client, data_mart_id, subscription_id):
    # ===========fairness  min100/hourly
    target = Target(
        target_type=TargetTypes.SUBSCRIPTION,
        target_id=subscription_id
    )
    parameters = {
        "features": [
            {"feature": "Sex",
             "majority": ['male'],
             "minority": ['female']
             },
            {"feature": "Age",
             "majority": [[26, 75]],
             "minority": [[18, 25]]
             }
        ],
        "favourable_class": ["No Risk"],
        "unfavourable_class": ["Risk"],
        "min_records": 100
    }
    thresholds = [{
        "metric_id": "fairness_value",
        "specific_values": [{
            "applies_to": [{
                "key": "feature",
                "type": "tag",
                "value": "Age"
            }],
            "value": 95
        },
            {
                "applies_to": [{
                    "key": "feature",
                    "type": "tag",
                    "value": "Sex"
                }],
                "value": 95
            }
        ],
        "type": "lower_limit",
        "value": 80.0
    }]
    fairness_monitor_instance_id = create_monitor(data_mart_id, target, parameters, thresholds, 'fairness',
                                                  wos_client)

    ### Get Fairness Monitor Instance
    wos_client.monitor_instances.show()

    ### Get run details
    runs = wos_client.monitor_instances.list_runs(fairness_monitor_instance_id, limit=1).result.to_dict()
    fairness_monitoring_run_id = runs["runs"][0]["metadata"]["id"]
    run_status = None
    while (run_status not in ["finished", "error"]):
        run_details = wos_client.monitor_instances.get_run_details(fairness_monitor_instance_id,
                                                                   fairness_monitoring_run_id).result.to_dict()
        run_status = run_details["entity"]["status"]["state"]
        print('run_status: ', run_status)
        if run_status in ["finished", "error"]:
            break
        time.sleep(10)

    ### Fairness run output
    wos_client.monitor_instances.get_run_details(fairness_monitor_instance_id,
                                                 fairness_monitoring_run_id).result.to_dict()

    wos_client.monitor_instances.show_metrics(monitor_instance_id=fairness_monitor_instance_id)


def explainability(wos_client, data_mart_id, subscription_id, sample_size=2):
    # ====================Explainability
    target = Target(
        target_type=TargetTypes.SUBSCRIPTION,
        target_id=subscription_id
    )
    parameters = {
        "enabled": True
    }
    explain_monitor_details = wos_client.monitor_instances.create(
        data_mart_id=data_mart_id,
        background_mode=False,
        monitor_definition_id=wos_client.monitor_definitions.MONITORS.EXPLAINABILITY.ID,
        target=target,
        parameters=parameters
    ).result

    scoring_ids = []

    for i in range(0, sample_size):
        n = random.randint(1, 100)
        scoring_ids.append(scoring_id + '-' + str(n))
    print("Running explanations on scoring IDs: {}".format(scoring_ids))

    explanation_types = ["lime", "contrastive"]
    result = wos_client.monitor_instances.explanation_tasks(scoring_ids=scoring_ids,
                                                            explanation_types=explanation_types).result
    print(result)

    ### Explanation tasks
    explanation_task_ids = result.metadata.explanation_task_ids
    return explanation_task_ids


def explanation_feature_map_plot(finished_explanations):
    explanation_features_map = {}
    for result in finished_explanations:
        print('\n>>>>>>>>>>>>>>>>>>>>>>\n')
        print(
            'explanation task: ' + str(result.metadata.explanation_task_id) + ', perturbed:' + str(
                result.entity.perturbed))
        if result.entity.explanations is not None:
            explanations = result.entity.explanations
            for explanation in explanations:
                if 'predictions' in explanation:
                    predictions = explanation['predictions']
                    for prediction in predictions:
                        predicted_value = prediction['value']
                        probability = prediction['probability']
                        print('prediction : ' + str(predicted_value) + ', probability : ' + str(probability))
                        if 'explanation_features' in prediction:
                            explanation_features = prediction['explanation_features']
                            for explanation_feature in explanation_features:
                                feature_name = explanation_feature['feature_name']
                                feature_weight = explanation_feature['weight']
                                if (feature_weight >= 0):
                                    feature_weight_percent = round(feature_weight * 100, 2)
                                    print(str(feature_name) + ' : ' + str(feature_weight_percent))
                                    task_feature_weight_map = {}
                                    task_feature_weight_map[
                                        result.metadata.explanation_task_id] = feature_weight_percent
                                    construct_explanation_features_map(feature_name, feature_weight_percent,
                                                                       explanation_features_map)
            print('\n>>>>>>>>>>>>>>>>>>>>>>\n')

    for key in explanation_features_map.keys():
        # plot_graph(key, explanation_features_map[key])
        values = explanation_features_map[key]
        plt.title(key)
        plt.ylabel('Weight')
        plt.bar(range(len(values)), values)
        plt.show()


def quality(wos_client, subscription_id, feedback_file_path):
    # =========================Quality monitoring and feedback logging
    target = Target(
        target_type=TargetTypes.SUBSCRIPTION,
        target_id=subscription_id
    )
    parameters = {
        "min_feedback_data_size": 90
    }
    thresholds = [
        {
            "metric_id": "area_under_roc",
            "type": "lower_limit",
            "value": .80
        }
    ]
    quality_monitor_instance_id = create_monitor(data_mart_id, target, parameters, thresholds, 'quality',
                                                 wos_client)

    ## Feedback logging
    ## Get feedback logging dataset ID
    feedback_dataset_id = None
    feedback_dataset = wos_client.data_sets.list(type=DataSetTypes.FEEDBACK,
                                                 target_target_id=subscription_id,
                                                 target_target_type=TargetTypes.SUBSCRIPTION).result
    feedback_dataset_id = feedback_dataset.data_sets[0].metadata.id
    if feedback_dataset_id is None:
        print("Feedback data set not found. Please check quality monitor status.")

    with open(feedback_file_path) as feedback_file:
        additional_feedback_data = json.load(feedback_file)

    wos_client.data_sets.store_records(feedback_dataset_id, request_body=additional_feedback_data,
                                       background_mode=False)
    wos_client.data_sets.get_records_count(data_set_id=feedback_dataset_id)

    run_details = wos_client.monitor_instances.run(monitor_instance_id=quality_monitor_instance_id,
                                                   background_mode=False).result
    wos_client.monitor_instances.show_metrics(monitor_instance_id=quality_monitor_instance_id)


def drift(wos_client, data_mart_id, training_file_path, subscription_id, score_method, drift_detection_input):
    # =======================Drift
    # Drift detection model generation
    df = pd.read_csv(training_file_path)
    ### Define the drift detection input

    ### Generate drift detection model
    filename = 'drift_detection_model.tar.gz'
    generating_drift_model(df, drift_detection_input, score_method)
    ### Upload the drift detection model to OpenScale subscription

    wos_client.monitor_instances.upload_drift_model(
        model_path=filename,
        archive_name=filename,
        data_mart_id=data_mart_id,
        subscription_id=subscription_id,
        enable_data_drift=True,
        enable_model_drift=True
    )

    ### Delete the existing drift monitor instance for the subscription
    remove_drift_monitor_for_subscription(wos_client, subscription_id)

    target = Target(
        target_type=TargetTypes.SUBSCRIPTION,
        target_id=subscription_id

    )
    parameters = {
        "min_samples": 100,
        "drift_threshold": 0.1,
        "train_drift_model": False,
        "enable_model_drift": True,
        "enable_data_drift": True
    }
    drift_monitor_instance_id = create_monitor(data_mart_id, target, parameters, {}, 'drift', wos_client)

    ### Drift run
    drift_run_details = wos_client.monitor_instances.run(monitor_instance_id=drift_monitor_instance_id,
                                                         background_mode=False)
    time.sleep(5)
    wos_client.monitor_instances.show_metrics(monitor_instance_id=drift_monitor_instance_id)


if __name__ == '__main__':
    WOS_CPD_TECHZONE = {
        "url": "https://services-uscentral.skytap.com:8586/",
        "username": "admin",
        "password": "password",
        "version": "3.5"
    }
    WOS_CPD_LUBAN = {
        "url": "https://zen-cpd-zen.apps.bj-prod-2.luban.cdl.ibm.com/",
        "username": "admin",
        "password": "password",
        "version": "3.5"
    }

    DB2_TECHZONE = {'table_name': 'GERMAN_CREDIT',
                    'schema_name': 'aiopenscale00',
                    'hostname': '10.1.1.1',
                    'username': 'admin',
                    'password': 'password',
                    'database_name': 'aiopenscale00'}
    DB2_LUBAN = {'table_name': 'GERMAN_CREDIT',
                 'schema_name': 'OPENSCALE',
                 'hostname': 'worker19.bj-prod-2.luban.cdl.ibm.com:30157',
                 'username': 'admin',
                 'password': 'password',
                 'database_name': 'aiopenscale00'}

    VM_SCORING_URL = 'http://169.62.165.235:8880/predict/'
    GBS_SCORING_URL = 'https://9.112.255.149/edi/service199/api'
    general_header = {
        "Content-Type": "application/json",
        "Custom_header_X": "Custom_header_X_value_Y"
    }
    GBS_request_header = {
        'Content-Type': 'application/json',
        'apikey': 'eyJhbGciOiJSUzI1NiIsInppcCI6IkdaSVAifQ.H4sIAAAAAAAAAC2MWwrCMBQF93K_cyFN0oZ0A-Iy8rhKtG1KHqKIezcWvw7DGeYNpTmYYU2BlrQXrOlOG15po2xrTBu6F_rFxrUAg1hKd72lH2QqqWVPp5zafg4wD5xBofyIng40hgE9d5iVNpyLaZQjg9aN4-01W_tOclL_71ZjzwclSWjucXBOorpYhXYUGjtpsloMRnj4fAE1qncluQAAAA.LcoU4zK_a1qq8mxZAc1iBD0wQ9XuHkgWvjOy5As4Ob1Zb2_uZ1MHPu_nYnoPnRi07EMC-p-kqP8ahjlATXvIElxhxkUXHJP11TbkAQGnwVgfYX_iH4VN8Pft8Ma8eIjpnZ-QH5XM9iYige9u9dFDR2RHcD94LPVHHwLs2CNMX_U'
    }
    # TECHZONE
    wos_config = WOS_CPD_TECHZONE
    db2_config = DB2_TECHZONE
    scoring_url = VM_SCORING_URL
    scoring_request_headers = general_header
    # # GBS
    # wos_config = WOS_CPD_LUBAN
    # db2_config = DB2_LUBAN
    # scoring_url = GBS_SCORING_URL
    # scoring_request_headers=GBS_request_header

    label_column = "Risk"
    model_type = "binary"

    training_file_path = "../data/german_credit_data_biased_training.csv"
    feedback_file_path = '../data/additional_feedback_data_v2.json'
    df = pd.read_csv(training_file_path)

    cols_to_remove = [label_column]

    payload_scoring = get_scoring_payload(df, cols_to_remove, 1)

    scoring_id = None

    wos_client = auth_cpd(wos_config)

    data_marts = wos_client.data_marts.list().result.data_marts
    if len(data_marts) == 0:
        raise Exception("Missing data mart.")
    data_mart_id = data_marts[0].metadata.id
    print('Using existing datamart {}'.format(data_mart_id))

    data_mart_details = wos_client.data_marts.list().result.data_marts[0]
    data_mart_details.to_dict()

    print(wos_client.service_providers.show())

    SERVICE_PROVIDER_NAME = "Custom ML Provider Demo - All Monitors"
    SERVICE_PROVIDER_DESCRIPTION = "Added by tutorial WOS notebook to showcase monitoring Fairness, Quality, Drift and Explainability against a Custom ML provider."

    remove_existing_service_provider(wos_client, SERVICE_PROVIDER_NAME)
    service_provider_id = add_service_provider(SERVICE_PROVIDER_NAME, SERVICE_PROVIDER_DESCRIPTION, )

    print('Data Mart ID : ' + data_mart_id)

    wos_client.subscriptions.show()

    SUBSCRIPTION_NAME = "Custom ML Subscription - All Monitors"
    remove_existing_subscription(wos_client, SUBSCRIPTION_NAME)

    feature_columns = ["CheckingStatus", "LoanDuration", "CreditHistory", "LoanPurpose", "LoanAmount",
                       "ExistingSavings",
                       "EmploymentDuration", "InstallmentPercent", "Sex", "OthersOnLoan", "CurrentResidenceDuration",
                       "OwnsProperty", "Age", "InstallmentPlans", "Housing", "ExistingCreditsCount", "Job",
                       "Dependents",
                       "Telephone", "ForeignWorker"]
    cat_features = ["CheckingStatus", "CreditHistory", "LoanPurpose", "ExistingSavings", "EmploymentDuration", "Sex",
                    "OthersOnLoan", "OwnsProperty", "InstallmentPlans", "Housing", "Job", "Telephone", "ForeignWorker"]

    asset_id = str(uuid.uuid4())
    asset_name = '[asset] ' + SUBSCRIPTION_NAME
    url = ''

    asset_deployment_id = str(uuid.uuid4())
    asset_deployment_name = asset_name
    asset_deployment_scoring_url = scoring_url

    scoring_endpoint_url = scoring_url

    subscription_details = wos_client.subscriptions.add(
        data_mart_id=data_mart_id,
        service_provider_id=service_provider_id,
        asset=Asset(
            asset_id=asset_id,
            name=asset_name,
            url=url,
            asset_type=AssetTypes.MODEL,
            input_data_type=InputDataType.STRUCTURED,
            problem_type=ProblemType.BINARY_CLASSIFICATION
        ),
        deployment=AssetDeploymentRequest(
            deployment_id=asset_deployment_id,
            name=asset_deployment_name,
            deployment_type=DeploymentTypes.ONLINE,
            scoring_endpoint=ScoringEndpointRequest(
                url=scoring_endpoint_url,
                request_headers=scoring_request_headers
            )
        ),
        asset_properties=AssetPropertiesRequest(
            label_column=label_column,
            probability_fields=["probability"],
            prediction_field="prediction",
            feature_fields=feature_columns,
            categorical_fields=cat_features,
            training_data_reference=TrainingDataReference(type="db2",
                                                          location=DB2TrainingDataReferenceLocation(
                                                              table_name=db2_config['table_name'],
                                                              schema_name=db2_config['schema_name']),
                                                          connection=DB2TrainingDataReferenceConnection.from_dict({
                                                              'hostname': db2_config['hostname'],
                                                              'username': db2_config['username'],
                                                              'password': db2_config['password'],
                                                              'database_name': db2_config['database_name']}))
        )
    ).result
    subscription_id = subscription_details.metadata.id
    print('Subscription ID: ' + subscription_id)

    time.sleep(5)
    payload_data_set_id = None
    payload_data_set_id = wos_client.data_sets.list(type=DataSetTypes.PAYLOAD_LOGGING,
                                                    target_target_id=subscription_id,
                                                    target_target_type=TargetTypes.SUBSCRIPTION).result.data_sets[
        0].metadata.id
    if payload_data_set_id is None:
        print("Payload data set not found. Please check subscription status.")
    else:
        print("Payload data set id:", payload_data_set_id)

    wos_client.subscriptions.get(subscription_id).result.to_dict()

    # send a request to the model before we configure OpenScale.
    # This allows OpenScale to create a payload log in the datamart with the correct schema, so it can capture data coming into and out of the model.

    payload_scoring = get_scoring_payload(df, cols_to_remove, no_of_records_to_score=100)
    scoring_response = custom_ml_scoring(scoring_url, payload_scoring)

    scoring_id = payload_logging(payload_scoring, scoring_response, wos_client, payload_data_set_id)
    print('scoring_id: ' + str(scoring_id))

    fairness(wos_client, data_mart_id, subscription_id)
    sample_size = 2
    # explanation_task_ids = explainability(wos_client, data_mart_id, subscription_id, sample_size=2)
    # finished_explanations = finish_explanation_tasks(wos_client, explanation_task_ids, sample_size=2)

    # explanation_task_ids = ['8e71821a-cb9c-453e-a08f-ab8e3395aa11', '39bc9469-c956-43b8-be83-1a344be39367']
    # finished_explanations = []
    # while len(finished_explanations) < sample_size:
    #     finished_explanations = finish_explanation_tasks(wos_client, explanation_task_ids, sample_size=2)

    # explanation_feature_map_plot(finished_explanations)
    quality(wos_client, subscription_id, feedback_file_path)
    drift_detection_input = {
        "feature_columns": feature_columns,
        "categorical_columns": cat_features,
        "label_column": label_column,
        "problem_type": model_type
    }
    drift(wos_client, data_mart_id, training_file_path, subscription_id, score,drift_detection_input)
