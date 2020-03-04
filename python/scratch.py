import argparse

parser = argparse.ArgumentParser("input three dummy lists")
parser.add_argument("-l1", "--list1", nargs="+", help="<required> Set flag", required=True)
parser.add_argument("-l2", "--list2", nargs="+", help="<required> Set flag", required=True)
parser.add_argument("-l3", "--list3", nargs="+", help="<required> Set flag", required=True)

args = parser.parse_args()

print("list1=",args.list1)
print("list2=",args.list2)
print("list3=",args.list3)
