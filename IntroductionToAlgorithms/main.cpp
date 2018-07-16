
#include <string>
#include "algorithms.h"
#include <time.h>
#include <iostream>
#include <Windows.h>
#include "Review.h"
#include <iomanip>
#include "secondPartLeetcode.h"
#include <bitset>
using namespace std;

int KMPMatch(const string &s, const string p)
{
	auto GetJumpTable = [](const string &pattern) {
		vector<int> ans(pattern.size(), -1);
		int q = -1;
		for (int i = 1; i < pattern.size(); ++i)
		{
			while (q > -1 && pattern[q + 1] != pattern[i])
				q = ans[q];
			if (pattern[q + 1] == pattern[i])
				++q;
			ans[i] = q;
		}
		return ans;
	};
	vector<int> pi = std::move(GetJumpTable(p));
	int q = -1;
	for (int i = 0; i < s.size(); ++i)
	{
		while (q > -1 && p[q + 1] != s[i])
			q = pi[q];
		if (p[q + 1] == s[i])
		{
			++q;
			if (q == p.size()-1)
				return i;
		}
	}
	return -1;

}

class Solution {
public:
	struct lesscmp
	{
		bool operator()(const ListNode* l1, const ListNode* l2)
		{
			return l1->val > l2->val;
		}
	};
	ListNode* mergeKLists(vector<ListNode*>& lists) {
		if (lists.size() == 0)
			return nullptr;
		priority_queue<ListNode*, vector<ListNode*>, lesscmp> pq;
		for (auto head : lists)
		{
			if (head != nullptr)
				pq.push(head);
		}
		auto iter = new ListNode(INT_MIN);
		auto temp = iter;
		while (!pq.empty())
		{
			auto cur = pq.top();
			pq.pop();
			if (cur->next != nullptr)
				pq.push(cur->next);
			iter->next = cur;
			iter = iter->next;
		}
		auto newhead = temp->next;
		delete temp;
		return newhead;
	}

};


double f(const string& num)
{
	double ans = 1;
	for (auto ch : num)
		ans *= (ch - '0');
	return ans;
}
bool DoubleEqual(double num1, double num2)
{
	return abs(num1 - num2) < 0.000001;
}
void decrease(string& num)
{
	int iter = num.size() - 1;
	while (iter>=0)
	{
		if (num[iter] == '0')
			num[iter--] = '9';
		else
		{
			--num[iter];
			break;
		}
	}
	if (num[0] == '0')
		num = num.substr(1);
}

int main()
{
	string instr;
	cin >> instr;
	double f_x = f(instr);
	decrease(instr);
	while (instr.size() != 0)
	{
		if (DoubleEqual(f(instr),f_x))
		{
			cout << instr << endl;
			break;
		}
		decrease(instr);
	}
	system("pause");
	return 0;
}
