#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <stack>
#include  <queue>
#include <set>
#include <iostream>
#include <algorithm>
using namespace std;
#define UINT unsigned int
struct ListNode
{
	int val;
	ListNode *next;
	ListNode(int x) :val(x), next(NULL) {};
};
struct Interval {
	int start;
	int end;
	Interval() : start(0), end(0) {}
	Interval(int s, int e) : start(s), end(e) {}
};
struct TreeNode
{
	int val;
	TreeNode *left;
	TreeNode *right;
	TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

TreeNode* BuildTree(const vector<string> &vec)
{
	if (vec.size() == 0)
		return NULL;
	vector<TreeNode*>  tmp(vec.size(), NULL);
	for (unsigned int i = 0; i < vec.size(); ++i)
	{
		if (vec[i] == "null")
			continue;
		else
		{
			tmp[i + 1] = new TreeNode(stoi(vec[i]));
		}
	}
	
	for (unsigned int i = 1; i <= tmp.size()/2; ++i)
	{
		if (tmp[i] != NULL)
		{
			tmp[i]->left = tmp[2 * i];
			tmp[i]->right = tmp[2 * i + 1];
		}
	}
	return tmp[1];
}
class DisJointSet
{
private:
	vector<int> theArray;
public:
	DisJointSet(int num)
	{
		theArray.resize(num);
		for (int i = 0; i < num; i++)
		{
			theArray[i] = -1;
		}
	}
	~DisJointSet()
	{
		theArray.clear();
	}
	void unionSet(int root1, int root2)//按大小求并
	{
		if (theArray[root1] <= theArray[root2])
		{
			theArray[root1] += theArray[root2];
			theArray[root2] = root1;
		}
		else
			unionSet(root2, root1);
	}
	int find(int root)
	{
		if (theArray[root]<0)
			return root;
		else
			return theArray[root] = find(theArray[root]);
	}
	int getDisJointSetNum()
	{
		int cnt = 0;
		for (unsigned int i = 0; i<theArray.size(); ++i)
		{
			if (theArray[i] < 0)
				cnt++;
		}
		return cnt;
	}
};
void printLongestPalindromeStr(string str, const vector<vector<int> > &dp_result, int i, int j, string& ans)
{
	if (i > j)
		return;
	if (i == j)
	{
		ans += str[i];
	}
	else
	{
		int tempM = max(dp_result[i + 1][j], dp_result[i][j - 1]);
		int temp1 = dp_result[i + 1][j];
		int temp2 = dp_result[i][j - 1];
		if (str[i]==str[j])
		{
			ans += str[i];
			printLongestPalindromeStr(str, dp_result, i + 1, j - 1, ans);
			ans += str[j];
		}
		else if (temp1 > temp2)
		{
			printLongestPalindromeStr(str, dp_result, i + 1, j, ans);
		}
		else
			printLongestPalindromeStr(str, dp_result, i, j - 1, ans);
	}
}
string getLongestPalindromeStr(string str)
{
	vector<int > aline(str.size(), 0);
	vector<vector<int> > dp(str.size(), aline);
	for (UINT i = 0; i < dp.size(); ++i)
		dp[i][i] = 1;
	for (UINT l = 2; l <= str.size(); ++l)
	{
		for (UINT i = 0; i <= str.size() - l; ++i)
		{
			int j = i + l - 1;
			if (str[i] == str[j])
				dp[i][j] = dp[i + 1][j - 1] + 2;
			else
			{
				dp[i][j] = max(dp[i + 1][j], dp[i][j - 1]);
			}
		}
	}
	string ans = "";
	printLongestPalindromeStr(str, dp, 0, str.size() - 1, ans);
	return ans;
}
void printLCS(const string &str1, const string &str2, const vector<vector<int> > & dp, int i, int j, string & ans)
{
	if (i == 0 || j == 0)
		return;
	if (str1[i] == str2[j])
	{
		printLCS(str1, str2, dp, i - 1, j - 1, ans);
		ans += str1[i];
	}
	else
	{
		if (dp[i][j - 1] > dp[i - 1][j])
			printLCS(str1, str2, dp, i, j - 1, ans);
		else
			printLCS(str1, str2, dp, i - 1, j, ans);
	}
}
string getLCS(string str1, string str2)
{
	vector<int> aline(str2.size()+1, 0);
	vector<vector<int> > dp(str1.size()+1, aline);
	for (UINT i = 1; i <= str1.size(); ++i)
	{
		for (UINT j = 1; j <= str2.size(); ++j)
		{
			dp[i][j] = str1[i] == str2[j] ? (dp[i - 1][j - 1] + 1) : max(dp[i - 1][j], dp[i][j - 1]);
		}
	}
	string ans = "";
	printLCS(str1, str2, dp, str1.size(), str2.size(), ans);
	return ans;
}

void printStringVec(const vector<string > &vec)
{
	for (unsigned int i = 0; i < vec.size(); ++i)
		cout << vec[i] << endl;
}

namespace Leetcode141
{
	
	class Solution {
	public:
		ListNode *detectCycle(ListNode *head) {
			if (head == NULL || head->next == NULL)
				return NULL;
			ListNode* slow = head;
			ListNode* fast = head;
			ListNode* entry = head;
			do
			{
				if (fast == NULL || fast->next == NULL)
					return NULL;
				slow = slow->next;
				fast = fast->next->next;
				if (slow == fast)
				{
					while (slow!=entry)
					{
						slow = slow->next;
						entry = entry->next;
					}
					return entry;
				}
			} while ((fast != slow));
			return NULL;
		}
	};
}
namespace Leetcode200
{
	class Solution1 {
	public:
		
		class isLandPart
		{
		public:
			int row;
			int col;
			bool used;
			isLandPart(int _row, int _col)
			{
				row = _row;
				col = _col;
				used = false;
			}
			static bool isNear(isLandPart part1, isLandPart part2)
			{
				bool isnear = ((part2.row == part1.row) && (part2.col == part1.col - 1 || part2.col == part1.col + 1)) || \
					((part2.col == part1.col) && (part2.row == part1.row - 1 || part2.row == part1.row + 1));
				return isnear;
			}
		};
		int numIslands(vector<vector<char>>& grid){
			vector<isLandPart> isLandParts;
		for (int i = 0; i<grid.size(); ++i)
		{
			for (int j = 0; j<grid[i].size(); ++j)
			{
				if (grid[i][j] == '1')
					isLandParts.push_back(isLandPart(i,j));
			}
		}
		DisJointSet  disj(isLandParts.size());
		for (int i = 0; i<isLandParts.size(); ++i)
		{
			for (int j = i + 1; j<isLandParts.size(); ++j)
			{
				if (isLandPart::isNear(isLandParts[i],isLandParts[j]))
				{
					int root1 = disj.find(i);
					int root2 = disj.find(j);
					if (root1 != root2)
						disj.unionSet(root1,root2);
				}
			}
		}
		return disj.getDisJointSetNum();
		}
	};
	class Solution2
	{
	public:
		void dfs(vector<vector<char> >& grid, vector<vector<bool>> &used, int i, int j)
		{
			if (!used[i][j] && grid[i][j] == '1')
			{
				used[i][j] = true;
				if (i + 1 < grid.size())
					dfs(grid, used, i + 1, j);
				if (j + 1 < grid[0].size())
					dfs(grid, used, i, j + 1);
				if (i - 1 >= 0)
					dfs(grid, used, i - 1, j);
				if (j - 1 >= 0)
					dfs(grid, used, i, j - 1);
			}
		}
		int  numIslands(vector<vector<char> >&grid)
		{
			if (grid.size() == 0)
				return 0;
			int cnt = 0;
			vector<bool> aline(grid[0].size(), false);
			vector<vector<bool>> used(grid.size(), aline);
			for (int i = 0; i < grid.size(); ++i)
			{
				for (int j = 0; j < grid[0].size(); ++j)
				{
					if (!used[i][j] && grid[i][j] == '1')
					{
						dfs(grid, used, i, j);
						++cnt;
					}
				}
			}
			return cnt;
		}
	};
}
namespace Leetcode282
{
		class Solution {
		public:
			void addOperators(vector<string>& ans, string t, string num, long long last, long long currentVal, int target)
			{
				if (num.size() == 0)
				{
					if (currentVal == target)
						ans.push_back(t);
					return;
				}
				for(int l = 1; l <= num.size(); ++l)
				{
					string firstNum = num.substr(0, l);
					if (firstNum.size()>1 &&firstNum[0] == '0')
						return;
					string nextNum = num.substr(l);
					if (t.size()>0)
					{
						addOperators(ans, t + "+" + firstNum, nextNum, stoll(firstNum), currentVal + stoll(firstNum), target);
						addOperators(ans, t + "-" + firstNum, nextNum, -stoll(firstNum), currentVal - stoll(firstNum), target);
						addOperators(ans, t + "*" + firstNum, nextNum, last*stoll(firstNum), currentVal - last + last*stoll(firstNum), target);
					}
					else
						addOperators(ans, firstNum, nextNum, stoll(firstNum), stoll(firstNum), target);
				}

			}
			vector<string> addOperators(string num, int target) {
				vector<string> ans;
				addOperators(ans, "", num, 0, 0, target);
				return ans;
			}
		};
}
namespace Leetcode216
{
	class Solution {
	public:
		void backtracking(vector<vector<int>> &ans, vector<int>& ans_part, int currentVal, int targetVal,int currentPos, int numUsed,int numLimit)
		{
			if (numUsed == numLimit)
			{
				if (currentVal == targetVal)
				{
					ans.push_back(ans_part);
				}
				return;
			}
			for (int i = currentPos; i <= 9; ++i)
			{
				if (currentVal + i > targetVal)
					return;
				ans_part.push_back(i);
				backtracking(ans, ans_part, currentVal + i, targetVal, i+1, numUsed + 1, numLimit);
				ans_part.pop_back();
			}
		}
		vector<vector<int>> combinationSum3(int k, int n) {
			vector<vector<int> > ans;
			vector<int > ans_part;
			backtracking(ans, ans_part, 0, n, 1, 0, k);
			return ans;
		}
	};
}
namespace Leetcode435
{
	
	
	class Solution {
	public:
		static bool cmpInterval(Interval & i1, Interval & i2)
		{
			return ((i1.end - i1.start) < (i2.end - i2.start));
		}
		bool isOverlap(Interval inter1, Interval inter2)
		{
			if (inter1.start <= inter2.start)
				return inter2.start < inter1.end;
			return isOverlap(inter2, inter1);
		}
		
		int eraseOverlapIntervals(vector<Interval>& intervals) {
			sort(intervals.begin(), intervals.end(), cmpInterval);
			vector<Interval>  legalIntervals;
			int cnt = 0;
			for (unsigned int i = 0; i < intervals.size(); ++i)
			{
				if (legalIntervals.size() == 0)
				{
					legalIntervals.push_back(intervals[i]);
					continue;
				}
				bool isLegal = true;
				for (unsigned int j = 0; j < legalIntervals.size(); ++j)
				{
					if (isOverlap(intervals[i], legalIntervals[j]))
					{
						isLegal = false;
						break;
					}
				}
				cnt += isLegal == true ? 0 : 1;
			}
			return cnt;
		}
	};
}

namespace Leetcode120
{
	class Solution {
	public:
		int minimumTotal(vector<vector<int>>& triangle) {
			if (triangle.size() == 1)
				return triangle[0][0];
			vector<vector<int> > dp;
			for (unsigned int i = 0; i < triangle.size(); ++i)
			{
				dp.push_back(vector<int>(i + 1, 0));
			}
			int res = INT_MAX;
			for (unsigned int i = 0; i < triangle.size(); ++i)
			{
				if (i == 0)
				{
					dp[0][0] = triangle[0][0];
					continue;
				}
				for (unsigned int j = 0; j < triangle[i].size(); ++j)
				{
					dp[i][j] = getMinDistAdjacent(i, j, dp)+triangle[i][j];
					if (i == dp.size() - 1 && dp[i][j] < res)
						res = dp[i][j];
				}
			}
			return res;
			
		}
	private:
		int getMinDistAdjacent(int i, int j, const vector<vector<int> > & dp)
		{
			
			int res = 0;
			if (i < 0)
				res = -1;
			else if (j - 1 < 0)
				res= dp[i - 1][j];
			else if (j == dp[i - 1].size())
				res= dp[i - 1][j - 1];
			else
			{
				res = min(dp[i - 1][j - 1], dp[i - 1][j]);
			}
			return res;
		}
	};
}

namespace Leetcode114
{
	
	
	class Solution {
	public:
		void flatten(TreeNode* root) {
			flat(root);
		}
		void printTree(TreeNode *t)
		{
			if (t == nullptr)
			{
				cout << "null,";
				return;
			}
			cout << t->val << ",";
			printTree(t->left);
			
			printTree(t->right);
		}
	private:
		void flat(TreeNode *t)
		{
			if (t == NULL)
				return;
			if (t->left == nullptr&&t->right == nullptr)
				return;
			else if (t->left == NULL)
				flat(t->right);
			else if (t->right == NULL)
			{
				flat(t->left);
				t->right = t->left;
				t->left = nullptr;
			}
			else
			{
				flat(t->left);
				TreeNode* tail = t->left;
				while (tail->right!=NULL)
				{
					tail = tail->right;
				}
				flat(t->right);
				TreeNode* temp = t->right;
				t->right = t->left;
				t->left = NULL;
				tail->right = temp;
			}
		}
	};
	
}

namespace Leetcode547
{
	int findCircleNum(vector<vector<int>>& M) {
		DisJointSet disj(M.size());
		for (unsigned int i = 0; i < M.size(); ++i)
		{
			for (unsigned int j = 0; j < i; ++j)
			{
				if (M[i][j] == 1)
				{
					int root1 = disj.find(i);
					int root2 = disj.find(j);
					if (root1 != root2)
						disj.unionSet(root1, root2);
				}
			}
		}
		return disj.getDisJointSetNum();
	}
}

namespace Leetcode94
{
	vector<int> inorderTraversal(TreeNode* root)
	{
		TreeNode * t=root;
		vector<int > res;
		stack<TreeNode* > funcStack;
		while (t!=NULL||!funcStack.empty())
		{
			if (t != NULL)
			{
				funcStack.push(t);
				t = t->left;
			}
			else
			{
				t = funcStack.top();
				funcStack.pop();
				res.push_back(t->val);
				t = t->right;
			}
		}
		return res;

	}
}  //智商压制
namespace Leetcode103
{
	
	struct HashFunc
	{
		std::size_t operator()(const TreeNode& key) const
		{
			using std::hash;

			return (hash<long>()((long)&key));
		}
	};
	struct EqualKey
	{
		bool operator()(const TreeNode& lhs, const TreeNode& rhs) const 
		{
			return (&lhs)==(&rhs);
		}
	};
	vector<vector<int >> bfs(TreeNode* t)
	{
		unordered_map<TreeNode, int, HashFunc, EqualKey> dict;
		vector<vector<int > > ans;
		vector<int> ans_part;
		queue<TreeNode* > q;
		q.push(t);
		dict[*t] = 0;
		int currentLevel = 0;
		while (!q.empty())
		{
			TreeNode* tmp = q.front();
			if(currentLevel!=dict[*tmp])
			{
				currentLevel = dict[*tmp];
				ans.push_back(ans_part);
				ans_part.clear();
			}
			ans_part.push_back(tmp->val);
			if (tmp->left != NULL)
			{
				q.push(tmp->left);
				dict[*(tmp->left)] = dict[*tmp] + 1;
			}
			if (tmp->right != NULL)
			{
				q.push(tmp->right);
				dict[*(tmp->right)] = dict[*tmp] + 1;
			}
			q.pop();
		}
		for (unsigned int i = 0; i < ans.size(); ++i)
		{
			if (i % 2 != 0)
				reverse(ans[i].begin(), ans[i].end());
		}
		return ans;
	}
}

namespace Leetcode24
{
	class Solution
	{
	public:
		ListNode* swapPairs(ListNode* head)
		{
			if (head == NULL)
				return NULL;
			if (head->next == NULL)
				return head;
			ListNode* nn = head->next->next;
			ListNode*  n = head->next;
			n->next = head;
			head->next = swapPairs(nn);
			return n;
		}
	};
}
namespace Leetcode34
{
	class Solution {
	public:
		vector<int> searchRange(vector<int>& nums, int target) {
			int left = 0;
			int right = nums.size()-1;
			vector<int> ans(2, -1);
			int targetBegin = -1;
			int targetEnd = -1;
			while (left<=right)
			{
				int mid = (left + right) / 2;
				if (nums[mid] == target)
				{
					targetBegin = mid;
					break;
				}
				else if (nums[mid] < target)
					left = mid + 1;
				else if (nums[mid] > target)
					right = mid - 1;
			}
			if (targetBegin == -1)
				return ans;
			for (int i = targetBegin; i >= 0; --i)
			{
				if (nums[i] == target)
					targetBegin = i;
				else
					break;
			}
			for (unsigned int i = targetBegin; i < nums.size(); ++i)
			{
				if (nums[i] == target)
					targetEnd = i;
				else
					break;
			}
			ans[0] = targetBegin;
			ans[1] = targetEnd;
			return ans;
		}
	};
	void testCode()
	{
		vector<int> nums = {1,1,1,1,1 };
		Solution s1;
		s1.searchRange(nums, 1);
		
	}
}

namespace Leetcode49
{
	class Solution {
	public:
		vector<vector<string>> groupAnagrams(vector<string>& strs) {
			map<string, multiset<string> > dict;
			for (string str : strs)
			{
				string temp = str;
				sort(temp.begin(), temp.end());
				dict[temp].insert(str);
			}
			vector<vector<string>>  anagrams;
			for (auto entry : dict)
			{
				vector<string>  anagram(entry.second.begin(), entry.second.end());
				anagrams.push_back(anagram);
			}
			return anagrams;

		}
	};
	void testCode()
	{
		Solution s1;
		vector<string> strs = { "eat","tea","tan","ate","nat","bat" };
		vector<vector<string >>  groupStrs = s1.groupAnagrams(strs);
		for (unsigned int i = 0; i < groupStrs.size(); ++i)
		{
			for (unsigned int j = 0; j < groupStrs[i].size(); ++j)
			{
				cout << groupStrs[i][j] << " ";
			}
			cout << endl;
		}

	}
}

namespace Leetcode56
{
	class Solution {
	public:
		vector<Interval> merge(vector<Interval>& intervals) {
			vector<Interval> ans;
			if (intervals.size() == 0)
				return ans;
			
			auto cmp = [](const Interval& i1, const Interval& i2)
			{
				return i1.start < i2.start;
			};
			sort(intervals.begin(), intervals.end(),cmp);
			Interval cur = intervals[0];
			for (unsigned int i = 1; i < intervals.size(); ++i)
			{
				if (intervals[i].start <=cur.end)
				{
					if (intervals[i].end >= cur.end)
						cur.end = intervals[i].end;
				}
				else
				{
					ans.push_back(cur);
					cur = intervals[i];
				}
			}
			ans.push_back(cur);
			return ans;
		}
	};
	void testCode()
	{
		vector<Interval>  intervals = { Interval(1,4),Interval(0,4) };
		Solution s;
		intervals=s.merge(intervals);
		for (unsigned i = 0; i < intervals.size(); ++i)
			cout << intervals[i].start << "," << intervals[i].end << " ";
	}
}

namespace Leetcode7
{
	class Solution {
	public:
		int reverse(int x) {
			long long res;
			bool sign = false;
			if (x < 0)
				sign = true;
			string numStr = to_string(x);
			string part = "";
			if (sign)
			{
				part = numStr.substr(1);
			}
			else
				part = numStr;
			std::reverse(part.begin(), part.end());
			int zeroEnd = 0;
			while (part[zeroEnd] == '0')
				++zeroEnd;
			string resStr = "";
			if (sign)
				resStr += "-";
			resStr += part.substr(zeroEnd);
			res = stoll(resStr);
			if (abs(res) > INT_MAX)
				return 0;
			return (int)res;
		}
	};
	void testCode()
	{
		int x = -123;
		Solution s1;
		cout << s1.reverse(x) << endl;
	}
}

namespace Leetcode62
{
	class Solution {
	public:
		int uniquePaths(int m, int n) {
			vector<int> aline(n, 0);
			vector<vector<int> > dp(m, aline);
			for (unsigned int i = 0; i<n; ++i)
				dp[0][i] = 1;
			for (unsigned int i = 0; i<m; ++i)
				dp[i][0] = 1;
			for (unsigned int i = 1; i<m; ++i)
			{
				for (unsigned int j = 1; j<n; ++j)
				{
					dp[i][j] = dp[i - 1][j] + dp[i][j-1];
				}
			}
			return dp[m - 1][n - 1];
		}
	};
}

namespace Leetcode102
{
		class Solution {
		public:
			vector<vector<int>> levelOrder(TreeNode* root) {
				queue < TreeNode*> q;
				vector<int>  ans_part;
				vector<vector<int> > ans;
				if (root == NULL)
					return ans;
				q.push(root);
				q.push(NULL);
				while (!q.empty())
				{
					TreeNode* t = q.front();
					q.pop();
					if (t != NULL)
					{
						ans_part.push_back(t->val);
						if (t->left != NULL)
							q.push(t->left);
						if (t->right != NULL)
							q.push(t->right);
					}
					else
					{
						ans.push_back(ans_part);
						ans_part.resize(0);
						if (q.size() > 0)
						{
							q.push(NULL);
						}
					}
				}
				return ans;
			}
		};
}

namespace Leetcode45
{
	class Solution {
	public:
		int jump(vector<int>& nums) {
			vector<int> dp(nums.size(), INT_MAX);
			dp[0] = 0;
			if (nums.size() == 1)
				return 0;
			for (unsigned int current = 1; current < nums.size(); ++current)
			{
				for (unsigned int i = 0; i < current; ++i)
				{
					if (i + nums[i] >= current)
					{
						dp[current] = min(dp[current], dp[i] + 1);
					}
				}
			}
			return dp[nums.size() - 1];
		}
	};
	//没AC啊cnm
}

namespace Leetcode109
{
	class Solution {
	public:
		TreeNode* sortedListToBST(ListNode* head) {
			return buildTree(head);
		}
		TreeNode* buildTree(ListNode* l)
		{
			if (l == NULL)
				return NULL;
			if (l->next == NULL)
				return new TreeNode(l->val);
			ListNode* slow = l;
			ListNode* fast = l;
			ListNode* midPrev = NULL;
			while (fast != NULL&&fast->next != NULL)
			{
				midPrev = slow;
				slow = slow->next;
				fast = fast->next->next;
			}
			midPrev->next = NULL;
			TreeNode* r = new TreeNode(slow->val);
			r->left = buildTree(l);
			r->right = buildTree(slow->next);
			return r;
		}
	};
}

namespace Leetcode73
{
	class Solution {
	public:
		void setZeroes(vector<vector<int>>& matrix) {
			bool row0HaveZero = false;
			bool col0HaveZero = false;
			for (unsigned int i = 0; i < matrix.size(); ++i)
			{
				if (matrix[i][0] == 0)
				{
					col0HaveZero = true;
					break;
				}
			}
			for(unsigned int j=0;j<matrix[0].size();++j)	
			{
				if (matrix[0][j] == 0)
				{
					row0HaveZero = 0;
					break;
				}
			}
			for (unsigned int i = 0; i < matrix.size(); ++i)
			{
				for (unsigned int j = 0; j < matrix[0].size(); ++j)
				{
					if (matrix[i][j] == 0)
					{
						matrix[0][j] = 0;
						matrix[i][0] = 0;
					}
				}
			}
			for (unsigned int i = 0; i < matrix.size(); ++i)
			{
				if (matrix[i][0] == 0)
					setRowZero(matrix, i);
			}
			for (unsigned int j = 0; j < matrix[0].size(); ++j)
			{
				if (matrix[0][j] == 0)
					setColZero(matrix, j);
			}
			if (row0HaveZero)
				setRowZero(matrix, 0);
			if (col0HaveZero)
				setColZero(matrix, 0);
		}
	private:
		void setRowZero(vector<vector<int>> & matrix, int row)
		{
			for (unsigned int j = 0; j < matrix[row].size(); ++j)
				matrix[row][j] = 0;
		}
		void setColZero(vector<vector<int>> & matrix, int col)
		{
			for (unsigned int i = 0; i < matrix.size(); ++i)
				matrix[i][col] = 0;
		}
	};
}



