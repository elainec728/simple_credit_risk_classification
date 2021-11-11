docker commands:
docker image build -t credit:latest -f Dockerfile .
docker run -p 0.0.0.0:8890:8000 -v /root/temp_guest/data:/credit/osdemo/data --name=osdemo -it credit:latest

mount ~/data dir to /credit/osdemo/data with traning data **german_credit_data_biased_training.csv** in it.
