#pragma once
#include <stack> 
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <map>
#include <queue>
#include <unordered_map>

using namespace std;

namespace STACK
{
	string toPostfixExpression(const string& str)
	{
		string ans = "";
		stack<char> opStack;
		unordered_map<char, int > inStackPriority;
		unordered_map<char, int >  inComingPriority;
		inStackPriority['#'] = 0;
		inStackPriority['('] = 1;
		inStackPriority['-'] = 3;
		inStackPriority['+'] = 3;
		inStackPriority['*'] = 5;
		inStackPriority['/'] = 5;
		inStackPriority[')'] = 6;

		inComingPriority['('] = 6;
		inComingPriority['/'] = 4;
		inComingPriority['*'] = 4;
		inComingPriority['+'] = 2;
		inComingPriority['-'] = 2;
		inComingPriority[')'] = 1;
		inComingPriority['#'] = 0;
		opStack.push('#');
		for (int i = 0; i < str.size(); ++i)
		{
			char curOp = str[i];
			char stackTopOp = opStack.top();
			if (isalpha(curOp))
				ans += curOp;
			else if (inStackPriority[stackTopOp] < inComingPriority[curOp])
				opStack.push(str[i]);
			else
			{
				while (inStackPriority[stackTopOp] > inComingPriority[curOp])
				{
					if(stackTopOp!='('&&stackTopOp!=')')
						ans.push_back(stackTopOp);
					opStack.pop();
					stackTopOp = opStack.top();
				}
				if (inStackPriority[stackTopOp] == inComingPriority[curOp])
				{
					if (stackTopOp != '('&&stackTopOp != ')')
						ans.push_back(stackTopOp);
					opStack.pop();
				}
				if (curOp != ')')
					opStack.push(curOp);
			}
		}
		while (opStack.top()!='#')
		{
			ans.push_back(opStack.top());
			opStack.pop();
		}
		return ans;
	}
}

namespace Calculation
{
	double calPi(int n)
	{
		auto sgnMinusOne = [](int n) {
			if (n % 2 == 0)
				return 1;
			return -1;
		};
		double ans = 0;
		int i = 0;
		while (i <= n)
		{
			ans += double(4) / (2 * i + 1)*sgnMinusOne(i);
			i++;
		}
		return ans;
	}
	int mod(int a, int b, int n) //重复平方法 a^b modn
	{
		int d = 1;
		auto getBits = [](int num) {
			vector<int> bits;
			while (num != 0)
			{
				bits.push_back(num % 2);
				num = num >> 1;
			}
			return bits;
		};
		auto bits = getBits(b);
		for (int i = bits.size() - 1; i >= 0; --i)
		{
			d = (d*d) % n;
			if (bits[i] == 1)
				d = (d*a) % n;

		}
		return d;
	}

	tuple<long long, long long, long long> ExtendedEuclid(long long a, long long b)
	{
		if (b == 0)
			return make_tuple(a, 1, 0);
		auto temp = ExtendedEuclid(b, a%b);
		long long d_ = get<0>(temp);
		long long x_ = get<2>(temp);
		long long y_ = get<1>(temp) - (a / b)*get<2>(temp);
		return make_tuple(d_, x_, y_);
	}
}


namespace Leetcode44
{
	bool match(char ch1, char ch2)
	{
		return ch1 == ch2 || ch2 == '?';
	}
	bool backtracking(const string& s, const string &p, int curS, int curP)
	{
		int lastMatchS = s.size();
		int lastMatchP = p.size();
		while (lastMatchP!=p.size())
		{
			if (curP >= p.size() || curS > s.size() || (curP < p.size() && curS < s.size() && !match(s[curS], p[curP])))
			{
				curS = lastMatchS + 1;
				curP = lastMatchP;
			}
			else if (curS == s.size() && curP == p.size())
				return true;
			else if (match(s[curS], p[curP]))
			{
				++curS; ++curP;
			}
			else if (p[curP] == '*')
			{
				while (curP < p.size() - 1 && p[curP + 1] == '*')
					++curP;
				if (curP == p.size() - 1)
					return true;
				while (curS < s.size())
				{
					if (match(s[curS], p[curP]))
					{
						lastMatchS = curS;
						lastMatchP = curP;
						++curP;
						break;
					}
					++curS;
				}
			}
		}
		return false;
	}
	bool isMatch(string s, string p) {
		return backtracking(s, p, 0, 0);
	}
	
}
