#pragma once
#include "Leetcode.h"

namespace Leetcode139   //厉害厉害
{
	class Solution {
	public:
		bool wordBreak(string s, vector<string>& wordDict) {
			unordered_set<string > dict(wordDict.begin(), wordDict.end());
			vector<bool> dp(s.size()+1, false);
			dp[0] = true;
			for (unsigned int i = 1; i <= s.size(); ++i)
			{
				for (int j = i - 1; j >= 0; --j)
				{
					if (dp[j])
					{
						string word = s.substr(j, i - j);
						if (dict.find(word) != dict.end())
							dp[i] = true;
					}
				}
			}
			for (auto num : dp)
			{
				cout << num << " ";
			}
			cout << endl;
			return dp[s.size()];
		}
	};

	void testCode()
	{
		Solution s1;
		string s = "catsanddog";
		vector<string>  wordDict = { "cat","cats" ,"and","sand","dog" };
		cout<<s1.wordBreak(s, wordDict) << endl;
	}
}

namespace Leetcode140
{
	class Solution {
	public:
		vector<string> wordBreak(string s, vector<string>& wordDict) {
			unordered_set<string > dict(wordDict.begin(), wordDict.end());
			vector<bool> dp(s.size() + 1, false);
			dp[0] = true;
			for (unsigned int i = 1; i <= s.size(); ++i)
			{
				for (int j = i - 1; j >= 0; --j)
				{
					if (dp[j])
					{
						string word = s.substr(j, i - j);
						if (dict.find(word) != dict.end())
							dp[i] = true;
					}
				}
			}
			vector<string> ans;
			if (dp[s.size()] == false)
				return ans;
			string strPart = "";
			int partEnd = s.size();
			int	partBegin = partEnd - 1;
			while (partEnd>0)
			{
				partBegin = partEnd - 1;
				while (dp[partBegin] != 1)
				{
					--partBegin;
				}

			}
		}
	};
}


class Trie
{
private:
	class TrieNode
	{
	public:
		char content;
		bool isEnd;
		vector<TrieNode*> children;
		TrieNode() :content(' '), isEnd(false), children(26, NULL) {};
		TrieNode(char ch) :content(ch), isEnd(false), children(26, NULL) {};
		TrieNode* subNode(char ch)
		{
			return children[ch - 'a'];
		}
		~TrieNode()
		{
			for (auto child : children)
				delete child;
		}
	};

	TrieNode* root;
public: 
	Trie()
	{
		root = new TrieNode();
	}
	~Trie()
	{
		delete root;
	}
	bool search(string key)
	{
		TrieNode* cur = root;
		for (auto ch : key)
		{
			cur = cur->subNode(ch);
			if (cur == nullptr)
				return false;
		}
		return cur->isEnd == true;
	}
	void insert(string word)
	{
		if (search(word)) return;
		TrieNode* cur = root;
		for (auto ch : word)
		{
			if (cur->subNode(ch) == nullptr)
			{
				cur->children[ch - 'a'] = new TrieNode(ch);
			}
			cur = cur->subNode(ch);
		}
		cur->isEnd = true;
	}
	 bool startsWith(string prefix)
	{
		 TrieNode* cur = root;
		 for (auto ch : prefix)
		 {
			 cur = cur->subNode(ch);
			 if (cur == nullptr)
				 return false;
		 }
		 return true;
	}
};

namespace Leetcode5
{
	class Solution
	{

	public:
			string longestPalindrome(string s) 
			{
			if (s.empty()) return "";
			if (s.size() == 1) return s;
			int min_start = 0, max_len = 1;
			for (int i = 0; i < s.size();)
			{
				if (s.size() - i <= max_len / 2) break;
				int j = i, k = i;
				while (k < s.size() - 1 && s[k + 1] == s[k]) ++k; // Skip duplicate characters.
				i = k + 1;
				while (k < s.size() - 1 && j > 0 && s[k + 1] == s[j - 1]) { ++k; --j; } // Expand.
				int new_len = k - j + 1;
				if (new_len > max_len) { min_start = j; max_len = new_len; }
			}
			return s.substr(min_start, max_len);
		}
	};
}

namespace Leetcode101
{
	class Solution {
	public:
		
		bool isSymmetric(TreeNode* root) {
			vector<vector<TreeNode*> > ans;
			PreOrder(ans, root, 0);
			for (unsigned int i = 0; i < ans.size(); ++i)
			{
				for (int left = 0, right = ans[i].size() - 1; left < ans[i].size() / 2; ++left, --right)
				{
					if (ans[i][left] != nullptr&&ans[i][right] != nullptr&&ans[i][left]->val != ans[i][right]->val)
						return false;
					else if (ans[i][left] != nullptr&&ans[i][right] == nullptr)
						return false;
					else if (ans[i][left] == nullptr&&ans[i][right] != nullptr)
						return false;
				}
			}
			return true;
		}
	private:
		void PreOrder(vector<vector<TreeNode*>> &ans, TreeNode* t,int depth)
		{
			if (ans.size() == depth)
				ans.push_back(vector<TreeNode*>());
			ans[depth].push_back(t);
			if (t->left != nullptr)
				PreOrder(ans, t->left, depth + 1);
			if (t->right != nullptr)
				PreOrder(ans, t->right, depth + 1);
		}
	};

}

namespace Leetcode207
{
	class Solution {
	public:
		bool canFinish(int numCourses, vector<pair<int, int>>& prerequisites) 
		{
			vector<pair<int, vector<int>> > graphic(numCourses, make_pair(0, vector<int>()));
			//pair的first表示节点的入度，pair的second表示该节点连接的节点
			for (unsigned int i = 0; i < prerequisites.size(); ++i)
			{
				int thisNode = prerequisites[i].first;
				int adjNode = prerequisites[i].second;
				graphic[thisNode].second.push_back(adjNode);
				++graphic[adjNode].first;
			}
			
			while (true)
			{
				int probVertex = getMinInVertex(graphic);
				if (graphic[probVertex].first == INT_MAX)
					break;
				if (graphic[probVertex].first != 0)
					return false;
				for (unsigned int i = 0; i < graphic[probVertex].second.size(); ++i)
				{
					int adjNode = graphic[probVertex].second[i];
					--graphic[adjNode].first;
				}
				graphic[probVertex].first = INT_MAX;
			}
			return true;
		}
	private:
		int getMinInVertex(vector<pair<int, vector<int>>>& graphic)
		{
			int minIndex =0;
			for (unsigned int i = 0; i < graphic.size(); ++i)
			{
				if (graphic[i].first < graphic[minIndex].first)
					minIndex=i;
			}
			return minIndex;
		}
	};
}

namespace Leetcode108
{
	class Solution {
	public:
		TreeNode* sortedArrayToBST(vector<int>& nums) {
			return sortedArrayToBST(nums,0,nums.size()-1);
		}
	private:
		TreeNode* sortedArrayToBST(vector<int>& nums, int left, int right)
		{
			if (left > right)
				return NULL;
			if (left == right)
				return new TreeNode(nums[left]);
			int mid = (left + right) / 2;
			TreeNode* t = new TreeNode(nums[mid]);
			t->left = sortedArrayToBST(nums, left, mid - 1);
			t->right = sortedArrayToBST(nums, mid + 1, right);
			return t;
		}
	};
}

namespace Leetcode124   //厉害厉害
{
	class Solution {
	public:
		int maxPathSum(TreeNode* root) {
			int ans = INT_MIN;
			maxPathSum(root, ans);
			return ans;
		}
	private:
		int maxPathSum(TreeNode* t, int& re)
		{
			if (t == nullptr)
				return 0;
			int leftMax = maxPathSum(t->left, re);
			int rightMax = maxPathSum(t->right, re);
			if (leftMax < 0) leftMax = 0;
			if (rightMax < 0)  rightMax = 0;
			if (leftMax + rightMax + t->val > re)  
			{
				re = leftMax + rightMax + t->val;
			}
			t->val += max(leftMax, rightMax); //从底部开始更新到每个节点的最长路径
			return  t->val;
		}
	};
	
}


namespace Leetcode239
{
	class Solution {
	public:
		vector<int> maxSlidingWindow(vector<int>& nums, int k) {
			deque<int> maxIndex;
			vector<int > ans;
			for (unsigned int i = 0; i < nums.size(); ++i)
			{
				if (maxIndex.empty())
					maxIndex.push_back(i);
				else if (maxIndex.size()+k-1==i)
				{
					ans.push_back(nums[maxIndex.front()]);
				}
				else if (nums[i] < nums[maxIndex.back()])
					maxIndex.push_back(i);
				else if (nums[i] > nums[maxIndex.back()])
				{
					while (!maxIndex.empty()&&nums[i] > nums[maxIndex.back()])
					{
						maxIndex.pop_back();
					}
				}
			}
		}
	};
}
namespace Leetcode215
{
	class Solution {
	public:
		int findKthLargest(vector<int>& nums, int k) {
			quickSelect(nums, 0, nums.size() - 1, nums.size() - k + 1);
			return nums[nums.size() - k];
		}
	private:
		int middle3(vector<int>& vec, int left, int right)  //三数中值分割,先把最小的放入left，再比较center和right

		{
			int center = (left + right) / 2;
			if (vec[center] < vec[left])
				swap(vec[left], vec[center]);
			if (vec[right] < vec[left])
				swap(vec[left], vec[right]);
			if (vec[right] < vec[center])
				swap(vec[center], vec[right]);
			swap(vec[center], vec[right - 1]);
			return vec[right-1];
		}
		void insertionSort(vector<int>& vec, int left, int right)
		{
			for (int i = left + 1; i <= right; ++i)
			{
				int tmp = vec[i];
				int j = i;
				for (; j > left&&vec[j - 1] > tmp; j--)
				{
					vec[j] = vec[j - 1];
				}
				vec[j] = tmp;
			}
		}
		void quickSelect(vector<int>& vec, int left, int right, int k)
		{
			if (left + 10 < right)
			{
				int p = middle3(vec, left, right);
				int i = left, j = right - 1;
				for (;;)
				{
					while (vec[++i] < p) {};
					while (vec[--j] > p) {};
					if (i < j)
						swap(vec[i], vec[j]);
					else
						break;
				
					
				}
				swap(vec[i], vec[right - 1]);
				if (k == i + 1)
					return;
				else if (k > i + 1)
					quickSelect(vec, i + 1, right, k);
				else
					quickSelect(vec, left, i - 1, k);
			}
			else
				insertionSort(vec, left, right);

			}
	};
	

}
namespace Leetcode240
{
	class Solution {
	public:
	
		bool searchMatrix(vector<vector<int>>& matrix, int target) {
			if (matrix.size() == 0)
				return false;
			int i = matrix.size() - 1;
			int j = 0;
			while (i>=0&&j<matrix[0].size())
			{
				if (matrix[i][j] == target)
					return true;
				else if (matrix[i][j] > target)
					--i;
				else
					++j;
			}
			return false;
		}
	};
	void testCode()
	{
		Solution s;
		vector<vector<int> > matrix = 
		{
			{1, 4, 7, 11, 15},
			{2, 5, 8, 12, 19},
			{3, 6, 9, 16, 22},
			{10, 13, 14, 17, 24},
			{18, 21, 23, 26, 30}
		};
		cout << s.searchMatrix(matrix, 55) << endl;
	}
}

namespace Leetcode74
{
	class Solution {
	public:
		inline pair<int, int> changeToRowCol(int index, vector<vector<int> >& matrix)
		{
			pair<int, int> res;
			res.first = index / matrix[0].size();
			res.second = index%matrix[0].size();
			return res;
		}
		bool searchMatrix(vector<vector<int>>& matrix, int target) {
			int left = 0, right = matrix.size()*matrix[0].size() - 1;
			while (left<=right)
			{
				int mid = (left + right) / 2;
				pair<int, int> rowcol = changeToRowCol(mid, matrix);
				if (matrix[rowcol.first][rowcol.second] == target)
					return true;
				else if (matrix[rowcol.first][rowcol.second] > target)
					right = mid - 1;
				else
					left = mid + 1;
			}
			return false;

		}
	};
}

namespace Leetcode14
{
	class Solution {
	public:
		string longestCommonPrefix(vector<string>& strs) {
			string prefix = "";
			if (strs.size() == 0)
				return prefix;
			int currentPos = 0;
			while (true)
			{
				if (currentPos >= strs[0].size())
					goto endLoop;
				char probCh = strs[0][currentPos];
				for (unsigned int i = 0; i < strs.size(); ++i)
				{
					if (currentPos >= strs[i].size())
						goto endLoop;
					if (strs[i][currentPos] != probCh)
						goto endLoop;
				}
				prefix += probCh;
				currentPos++;
			}
			
		endLoop:
			return prefix;
		}
	};
}

namespace Leetcode210
{
	class Solution {
	public:
		vector<int> findOrder(int numCourses, vector<pair<int, int>>& prerequisites) {
			vector<int> probOrder;
			vector<pair<int, vector<int>> > graphic(numCourses, make_pair(0, vector<int>()));
			//pair的first表示节点的入度，pair的second表示该节点连接的节点
			for (unsigned int i = 0; i < prerequisites.size(); ++i)
			{
				int thisNode = prerequisites[i].second;
				int adjNode = prerequisites[i].first;
				graphic[thisNode].second.push_back(adjNode);
				++graphic[adjNode].first;
			}
			while (true)
			{
				int probVertex = getMinInVertex(graphic);
				if (graphic[probVertex].first == INT_MAX)
					break;
				if (graphic[probVertex].first != 0)
					return vector<int>();
				probOrder.push_back(probVertex);
				for (unsigned int i = 0; i < graphic[probVertex].second.size(); ++i)
				{
					int adjNode = graphic[probVertex].second[i];
					--graphic[adjNode].first;
				}
				graphic[probVertex].first = INT_MAX;
			}
			return probOrder;
		}
	private:
		int getMinInVertex(vector<pair<int, vector<int>>>& graphic)
		{
			int minIndex = 0;
			for (unsigned int i = 0; i < graphic.size(); ++i)
			{
				if (graphic[i].first < graphic[minIndex].first)
					minIndex = i;
			}
			return minIndex;
		}
	};
}

namespace Leetcode212
{
	class Solution {
	public:
		vector<string> findWords(vector<vector<char>>& board, vector<string>& words) {
			unordered_map<string, bool> dict;
			for (int k = 0; k < words.size(); ++k)
			{
				dict[words[k]] = false;
				for (int i = 0; i < board.size(); ++i)
				{
					for (int j = 0; j < board[i].size(); ++j)
					{
						if (board[i][j] == words[k][0] && dict[words[k]] == false)
						{
							bool find = false;
							dfs(board, i, j, 0, words[k], find);
							dict[words[k]] = find;
						}
					}
				}
			}
			vector<string> ans;
			for (auto entry : dict)
			{
				if (entry.second == true)
					ans.push_back(entry.first);
			}
			return ans;
		}
	private:
		void dfs(vector<vector<char>> &board, int cur_X, int cur_Y, int cur_Pos, const string & target, bool& find)
		{
			if (target[cur_Pos] == board[cur_X][cur_Y])
			{
				if (cur_Pos == target.size() - 1)
				{
					find = true;
					return;
				}
				char ch = board[cur_X][cur_Y];
				board[cur_X][cur_Y] =' ';
				if(cur_X-1>=0&&board[cur_X-1][cur_Y]!=' ')
					dfs(board, cur_X - 1, cur_Y, cur_Pos + 1, target, find);
				if (cur_X + 1<board.size()&& board[cur_X + 1][cur_Y] != ' ')
					dfs(board, cur_X + 1, cur_Y, cur_Pos + 1, target, find);
				if (cur_Y - 1 >= 0 && board[cur_X][cur_Y-1] != ' ')
					dfs(board, cur_X, cur_Y - 1, cur_Pos + 1, target, find);
				if (cur_Y + 1 < board[0].size() && board[cur_X][cur_Y+1] != ' ')
					dfs(board, cur_X, cur_Y + 1, cur_Pos + 1, target, find);
				board[cur_X][cur_Y] = ch;
			}
		}
	};
	void testCode()
	{
		vector<vector<char> > board = { {'a','b' },{'a','a'} };
		vector<string> words = { "aaba" };
		Solution s;
		vector<string> t= s.findWords(board, words);
		printStringVec(t);
	}
}
namespace Leetcode211
{
	class WordDictionary {
	private:
		class TrieNode
		{
		public:
			char content;
			bool isEnd;
			vector<TrieNode*> children;
			TrieNode() :content(' '), isEnd(false), children(26, NULL) {};
			TrieNode(char ch) :content(ch), isEnd(false), children(26, NULL) {};
			TrieNode* subNode(char ch)
			{
				return children[ch - 'a'];
			}
			~TrieNode()
			{
				for (auto child : children)
					delete child;
			}
		};
		TrieNode* root;
	public:
		/** Initialize your data structure here. */
		WordDictionary() {
			root = new TrieNode();
		}

		/** Adds a word into the data structure. */
		void addWord(string word) {
			TrieNode* cur = root;
			for (auto ch : word)
			{
				if (cur->subNode(ch) == nullptr)
					cur->children[ch-'a'] = new TrieNode(ch);
				cur = cur->subNode(ch);
			}
			cur->isEnd = true;
		}

		/** Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter. */
		bool search(string word) {
			 

		}
		bool search(string word, TrieNode* t)
		{
			if (word.size() > 0 && word[0] == '.')
			{
				bool find = false;
				for (int i = 0; i < 26; ++i)
				{
					if (t->children[i]!=nullptr)
						find = search(word.substr(1), t->children[i]);
					if (find)
						break;
				}
				return find;
			}
			else if(word.size()>0&&word[0]!='.')
			{
				if (t->subNode(word[0]) != nullptr)
					return search(word.substr(1), t->subNode(word[0]));
				else
					return false;
			}
			return t->isEnd;
		}
	};
}

namespace Leetcode230
{
	class Solution {
	public:
		int kthSmallest(TreeNode* root, int k) {
			int cur = 0;
			int result = 0;
			kthSmallest(root, k, cur, result);
			return result;
		}
	private:
		void kthSmallest(TreeNode* t, int k, int &cur, int& result)
		{
			if (cur <= k&&t!=nullptr)
			{
				kthSmallest(t->left, k, cur, result);
				if (cur == k)
				{
					result = t->val;
					cur = k + 1;
					return;
				}
				++cur;
				kthSmallest(t->right, k, cur, result);
			}
		}
	};
}

namespace Leetcode287
{
	class Solution {
	public:
		int findDuplicate(vector<int>& nums) {
			int left = 1;
			int n= nums.size() - 1;
			int right = n;
			while (left<right)
			{
				int mid = (left + right) / 2;
				int cnt = 0;
				for (auto num : nums)
				{
					if (num <= mid)
						++cnt;
				}
				if (cnt > mid)  //如果小于等于mid的数大于mid个,那么在[left,mid]中有可能的数
					right = mid ;
				else            //如果小于等于mid的数小于mid个,那么只可能在[mid+1,right]中有可能的数
					left = mid+1;
			
			}
			return left;
		}
	};
}
namespace Leetcode295
{
	class MedianFinder {
	public:
		/** initialize your data structure here. */
		MedianFinder() {

		}

		void addNum(int num) {
			numSet.insert(num);
			if (numSet.size() == 1)
			{
				midIters = make_pair(numSet.begin(), numSet.begin());
				return;
			}
			else if (numSet.size() == 2)
			{
				midIters = make_pair(numSet.begin(), ++numSet.begin());
				return;
			}
			
			if (numSet.size() % 2 == 0)
			{
				
				if (num >= *(midIters.first) && num <= *(midIters.second))
				{
					midIters.second = midIters.first = ++midIters.first;
				}
				else if (num > (*midIters.second))
				{
					midIters.first = midIters.second;
				}
				else
					midIters.second = midIters.first;
			}
			else
			{
				if (num >= *(midIters.first))
				{
					auto old = midIters.first;
					midIters.second = ++old;
				}
				else
				{
					auto old = midIters.first;
					midIters.first = --old;
				}

			}
		}

		double findMedian() {
			return (1.0*(*(midIters.first) + *(midIters.second))) / 2;
		}
	private:
		multiset<int> numSet;
		pair<decltype(numSet.begin()), decltype(numSet.begin())> midIters;
	};

	
}

namespace Leetcode33
{
	class Solution {
	public:
		int search(vector<int>& nums, int target) {
			int pivot = getRotatePivot(nums);
			int left = 0;
			int right =nums.size()-1;
			int n = nums.size();
			while (left<=right)
			{
				int mid = (left + right) / 2;
				int tMid = getRotatedIndex(mid, pivot, n);
				int tMidNum = nums[tMid];
				if(tMidNum==target)
					return tMid;
				else if (tMidNum > target)
					right = mid - 1;
				else
					left = mid + 1;					
			}
			return -1;
		}
	private:
		int getRotatedIndex(int index, int rotatePivot, int n)
		{
			return (index + rotatePivot) % n;
		}
		int getRotatePivot(vector<int>& nums)
		{
			int left = 0;
			int right = nums.size() - 1;
			while (left<right)
			{
				int mid = (left + right) / 2;
				if (nums[mid] > nums[right])
					left = mid+1;
				else
					right = mid;
			}
			return (left+right)/2;
		}
	};
	void testCode()
	{
		Solution s1;
		vector<int> nums = { 1 };
		cout << s1.search(nums, 0) << endl;
	}
}

namespace Leetcode49
{
	bool isAna(const string& s1, const string &s2)
	{
		int count[26] = { 0 };
		for (int i = 0; i<s1.size(); ++i)
			++count[s1[i] - 'a'];
		for (int i = 0; i<s2.size(); ++i)
			--count[s2[i] - 'a'];
		for (int i = 0; i<26; ++i)
			if (count[i] != 0)
				return false;
		return true;
	}
	void vecErase(vector<string>& strs, int index)
	{
		swap(strs[index], strs.back());
		strs.pop_back();
	}
	vector<vector<string>> groupAnagrams(vector<string>& strs) {
		vector<vector<string> >  ans;
		vector<string> anspart;
		vector<int> deleteIndices;
		for (int i = 0; i<strs.size(); ++i)
		{
			anspart.push_back(strs[i]);
			for (int j = i + 1; j<strs.size(); ++j)
			{
				if (isAna(strs[i], strs[j]))
				{
					anspart.push_back(strs[j]);
					deleteIndices.push_back(j);
				}
			}
			ans.push_back(anspart);
			anspart.clear();
			for_each(deleteIndices.begin(), deleteIndices.end(),
				[&](int index) {vecErase(strs, index); });
			
		}
		return ans;
	}
}