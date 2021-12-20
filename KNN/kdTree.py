"""
coding: utf-8
author: tianqi
email: tianqixie98@gmail.com
"""

class BinTreeNode(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def DLR(root):
    """
    :desc 先序遍历
    :param root:
    :return:
    """
    if not root:
        return
    else:
        print(root.val)
        DLR(root.left)
        DLR(root.right)

def LDR(root):
    """
    :desc 中序遍历
    :param root:
    :return:
    """
    if not root:
        return
    else:
        DLR(root.left)
        print(root.val)
        DLR(root.right)

def LRD(root):
    """
    :desc 后序遍历
    :param root:
    :return:
    """
    if not root:
        return
    else:
        DLR(root.left)
        DLR(root.right)
        print(root.val)

if __name__ == '__main__':
    root = BinTreeNode(0)
    root.left = BinTreeNode(1)
    root.right = BinTreeNode(2)
    DLR(root)