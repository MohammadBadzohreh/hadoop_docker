#!/usr/bin/env python3
import sys
import os
import shutil
from datetime import datetime
from pyspark import SparkConf, SparkContext

def parse_line(line):
    parts = line.strip().split('\t')
    if len(parts) < 1 or parts[0] == '':
        return None
    user = parts[0].strip()
    friends = []
    if len(parts) == 2 and parts[1].strip() != "":
        for f in parts[1].split(','):
            ff = f.strip()
            if ff != "":
                friends.append(ff)
    return user, friends

def generate_pairs(user_friends):
    user, friends = user_friends
    results = []
    # Direct markers
    for f in friends:
        try:
            u_int = int(user); f_int = int(f)
            key = (user, f) if u_int < f_int else (f, user)
        except:
            key = (user, f) if user < f else (f, user)
        results.append((key, "DIRECT"))
    # Mutual contributions
    n = len(friends)
    for i in range(n):
        fi = friends[i]
        for j in range(i+1, n):
            fj = friends[j]
            try:
                fi_int = int(fi); fj_int = int(fj)
                key = (fi, fj) if fi_int < fj_int else (fj, fi)
            except:
                key = (fi, fj) if fi < fj else (fj, fi)
            results.append((key, 1))
    return results

def sum_counts(a, b):
    if a == "DIRECT" or b == "DIRECT":
        return "DIRECT"
    else:
        return a + b

def select_topN(iterable, N=10):
    lst = list(iterable)
    def sort_key(x):
        uid, cnt = x
        try:
            uid_int = int(uid)
        except:
            uid_int = float('inf')
        return (-cnt, uid_int)
    lst.sort(key=sort_key)
    return [uid for uid, cnt in lst[:N]]

def delete_local_path(local_path):
    if os.path.exists(local_path):
        try:
            shutil.rmtree(local_path)
            print(f"Deleted existing output path: {local_path}", file=sys.stderr)
            return True
        except Exception as e:
            print(f"Could not delete output path {local_path}: {e}", file=sys.stderr)
            return False
    else:
        return True

def choose_output_path(out_path_arg):
    # Only handles file:// local paths; for HDFS URIs we just return as-is
    if not out_path_arg.startswith("file://"):
        return out_path_arg
    local_base = out_path_arg[7:]
    ok = delete_local_path(local_base)
    if ok:
        return out_path_arg
    else:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        new_local = os.path.join(local_base, timestamp)
        print(f"Will write to subdirectory: {new_local}", file=sys.stderr)
        return "file://" + new_local

def main():
    if len(sys.argv) != 3:
        print("Usage: python people_you_might_know.py <inputPath> <outputPath>", file=sys.stderr)
        sys.exit(1)
    input_path = sys.argv[1]
    output_path_arg = sys.argv[2]
    output_path = choose_output_path(output_path_arg)
    print(f"Using output path: {output_path}", file=sys.stderr)

    conf = SparkConf().setAppName("PeopleYouMightKnow").set("spark.ui.port", "4040")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    # If HDFS path, check connectivity
    if input_path.startswith("hdfs://"):
        try:
            hadoop_conf = sc._jsc.hadoopConfiguration()
            fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
            path = sc._jvm.org.apache.hadoop.fs.Path(input_path)
            if not fs.exists(path):
                print(f"ERROR: HDFS path does not exist or not reachable: {input_path}", file=sys.stderr)
                sys.exit(1)
            else:
                print(f"HDFS connectivity OK: found input path: {input_path}", file=sys.stderr)
        except Exception as e:
            print(f"ERROR: Unable to connect to HDFS or check path {input_path}: {e}", file=sys.stderr)
            sys.exit(1)

    lines = sc.textFile(input_path)
    userFriends = lines.map(parse_line).filter(lambda x: x is not None)
    pairMarkers = userFriends.flatMap(generate_pairs)
    aggregated = pairMarkers.reduceByKey(sum_counts)
    candidates = aggregated.filter(lambda kv: isinstance(kv[1], int) and kv[1] > 0)
    perUser = candidates.map(lambda kv: (kv[0][0], (kv[0][1], kv[1])))
    grouped = perUser.groupByKey()
    recommendations = grouped.mapValues(lambda itr: select_topN(itr, N=10))
    output_rdd = recommendations.map(lambda uc: f"{uc[0]}\t{','.join(uc[1])}")
    output_rdd.saveAsTextFile(output_path)
    print(f"Saved recommendations to {output_path}", file=sys.stderr)
    sc.stop()

if __name__ == "__main__":
    main()
