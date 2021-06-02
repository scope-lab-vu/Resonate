import glob
import os
import rosbag
import concurrent.futures
import multiprocessing
import time

USE_MULTIPROCESSING = True
NUM_WORKERS = 8
ROS_TOPIC = "/iver0/pose_gt"


def read_datafile(file_path):
    # Read messages from the bag file
    bag = rosbag.Bag(file_path)
    data = []
    for topic, msg, timestamp in bag.read_messages(ROS_TOPIC):
        # Read some data from the message
        data.append(msg.pose.pose.position.x)
    return data


if __name__ == "__main__":
    # DATA_FILE = "example_data/no_fault.csv"
    # DATA_FILE = "estimation_data/c_0.0_p_0.0_d_0.0/fm0.csv"
    # data_files = [DATA_FILE]

    DATA_DIR = "estimation_data"
    # DATA_DIR = "estimation_data/No-Faults/static/run3"

    # # Data with faults but without thruster reallocation
    # data_file_names = glob.glob(os.path.join(DATA_DIR, "**", "*.bag"), recursive=True)
    # data_file_names = [x for x in data_file_names if "/ais/" not in x]
    # data_file_names = [x for x in data_file_names if "/Fault-Reallocation/" not in x]
    # data_file_names = [x for x in data_file_names if "/Emergency-brake/" not in x]

    # # Data with faults AND thruster reallocation
    # data_file_names = glob.glob(os.path.join(DATA_DIR, "Fault-Reallocation/th0/**", "*.bag"), recursive=True)

    # Emergency Stop data
    # data_file_names = glob.glob(os.path.join(DATA_DIR, "No-Faults/static/Emergency-brake/**", "*.bag"), recursive=True)
    data_file_names = glob.glob(os.path.join(DATA_DIR, "ebrake/**", "*.bag"), recursive=True)

    # Get file sizes and compute average
    file_sizes = []
    for file in data_file_names:
        file_sizes.append(os.path.getsize(file))
    total_file_size_mb = sum(file_sizes) / float(1024 * 1024)
    avg_file_size_mb = total_file_size_mb / len(file_sizes)

    # Start time
    start_time = time.clock_gettime(time.CLOCK_MONOTONIC)

    # TODO: Multithreading did not produce large speedup like expected. Why not? No obvious HW bottleneck.

    # Read Datafiles either using multiple threads or multiple processes
    if USE_MULTIPROCESSING:
        with multiprocessing.Pool(NUM_WORKERS) as pool:
            # Start the read operations asynchronously
            completion_count = 0
            dataset = []
            for data in pool.imap_unordered(read_datafile, data_file_names):
                # Store data to dataset
                if data is not None:
                    dataset.append(data)

                # Print update every 10 bag files
                completion_count += 1
                if completion_count % 10 == 0:
                    print("Processed %d/%d bag files..." % (completion_count, len(data_file_names)))
        print("Finished processing bag files.")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Start the read operations and mark each future with its filename
            future_to_file = {executor.submit(read_datafile, file): file for file in data_file_names}

            # As each thread completes, store the resulting datafile and report progress
            completion_count = 0
            dataset = []
            for future in concurrent.futures.as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (file, exc))
                else:
                    if data is not None:
                        dataset.append(data)

                # Print update every 10 bag files
                completion_count += 1
                if completion_count % 10 == 0:
                    print("Processed %d/%d bag files..." % (completion_count, len(future_to_file)))
        print("Finished processing bag files.")

    # Compute time statistics
    end_time = time.clock_gettime(time.CLOCK_MONOTONIC)
    dt = end_time - start_time
    print("Number of workers: %d" % NUM_WORKERS)
    print("Bag file count, total size (MB), avg size (MB): %d, %.1f, %.1f" % (len(data_file_names), total_file_size_mb, avg_file_size_mb))
    print("Total time (s): %.3f" % dt)
    print("Time per file (s): %.3f" % (dt/float(len(data_file_names))))
    print("Throughput (MB/s): %.1f" % (total_file_size_mb / dt))

