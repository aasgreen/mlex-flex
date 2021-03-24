# create a job to handle segmenting
from kubernetes import client, config, watch
def create_job_object(job_name):
    container = client.V1Container(
            name="seg",
            image="aasgreen/mlexflex-ml-scipy-demo-deploy",
            command=["python", "unsupe-kmeans.py", "/data/in/meat-label.png"],
            image_pull_policy = "Always",
            volume_mounts=[client.V1VolumeMount(mount_path="/data", name="perm-data")]
            )
    volume = client.V1Volume(
          name="perm-data",
            persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(claim_name = "perm")
            )
    template = client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(labels={"app":"seg"}),
                spec=client.V1PodSpec(restart_policy="Never", volumes = [volume], containers=[container]))

    #create deploy spec
    spec = client.V1JobSpec(
            template=template,
            backoff_limit=4)

    #create job
    job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(name=job_name),
            spec=spec)
    return job

def create_job(api_instance, job):
    print('create_job: launching job')
    api_response = api_instance.create_namespaced_job(
            body = job,
            namespace='default')
    print("Job created. status={}".format(api_response.status))

def main(job_name):
    print('connecting to cluster')
    config.load_incluster_config()
    batch_v1 = client.BatchV1Api()
    print('connected to cluster: {}'.format(batch_v1.get_api_resources()))

    job = create_job_object(job_name)
    print('created job: {}'.format(job))

    create_job(batch_v1, job)
    print('launched job')
    try:
        new_api = client.CoreV1Api()
        print('connected to new api: {}'.format(new_api))
    # get pod name of job for logs
        job_def = batch_v1.read_namespaced_job(name=job_name, namespace='default')
        print(job_def)
        controller_uid = job_def.metadata.labels['controller-uid']
        print('cont {}'.format(controller_uid))
    except Exception as e:
        print(e)

    core_v1 = client.CoreV1Api()

    pod_label_selector = "controller-uid="+controller_uid
    print('pod {}'.format(pod_label_selector))
    pods_list = batch_v1.list_namespaced_job(namespace='default', label_selector=pod_label_selector, timeout_seconds = 10)
    print('finished')
    return batch_v1, job

if __name__ == '__main__':
    job_name = 'test-1'
    main(job_name)
