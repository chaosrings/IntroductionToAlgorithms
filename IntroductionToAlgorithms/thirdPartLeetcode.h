#pragma once
#include "DataStructures.h"
#include "Review.h"
#include "algorithms.h"

namespace Leetcode110
{
	bool isBalanced(TreeNode* root) {
		int balance = 1;
		height(root, balance);
		return balance;
	}
	int height(TreeNode* t,int& balance)
	{
		if(t!=nullptr&&balance)
		{
			int lheight = height(t->left,balance)+1;
			int rheihgt = height(t->right,balance)+1;
			if (abs(lheight - rheihgt) > 1)
				balance = false;
		}
		return 0;
	}
}
namespace Leetcode147
{
	ListNode* insertionSortList(ListNode* head) {
		ListNode* iter = nullptr;
		for (ListNode* i = head; i != nullptr; i = i->next)
		{
			iter = head;
			int key = i->val;
			while (iter->val < key)
				iter = iter->next;
			swap(i->val, iter->val);
		}
		return head;
	}
}

namespace TopKFrequentElements
{
	struct pqstruct
	{
		int value;
		int count;
		pqstruct(int _value = 0, int _count = 0) :value(_value), count(_count) {}
		friend bool operator<(const pqstruct &lhs,const pqstruct & rhs)   //自定义优先队列重载<应写成友元
		{
			return lhs.count < rhs.count;
		}
	};
	vector<int> topKFrequent(vector<int>& nums, int k) {
		unordered_map<int, int> dict;
		priority_queue<pqstruct> pq;
		vector<int> ans;
		for_each(nums.begin(), nums.end(), [&dict](int num) {
				++dict[num];
		});	
		for_each(dict.begin(), dict.end(), [&pq](pair<int, int> entry) {
			pq.push(pqstruct(entry.first, entry.second));
		});
		while (k--)
		{
			auto cur = pq.top();
			pq.pop();
			ans.push_back(cur.value);
		}
		return ans;
	}

}

