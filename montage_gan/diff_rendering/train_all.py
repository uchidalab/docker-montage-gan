# scripts = ["main_sigmoid.py", "main_tanh.py"]
scripts = ["main_tanh.py", "main_sigmoid.py"]
for s in scripts:
    print(f"=== Executing {s} ===")
    exec(open(s).read())
