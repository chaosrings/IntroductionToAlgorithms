#pragma once
#include <vector>
#include <queue>
#include <string>
namespace DataStructures
{
	using namespace std;
	struct ListNode
	{
		int val;
		ListNode *next;
		ListNode(const int& x=-1,ListNode* _next=nullptr) :val(x), next(_next) {};
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
	struct Edge
	{
		int value;
		string adjacentVertex1;
		string adjacentVertex2;
		Edge() { value = 0, adjacentVertex1 = "", adjacentVertex2 = ""; };
		Edge(string adjV1, string adjV2, int _value) :adjacentVertex1(adjV1), adjacentVertex2(adjV2), value(_value) {};
		bool operator<(const Edge & rhs) const   //this的优先级<rhs的优先级,因为this的value>rhs的value
		{
			return this->value > rhs.value;
		}
		bool operator>(const Edge & rhs) const
		{
			return this->value < rhs.value;
		}
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

		for (unsigned int i = 1; i <= tmp.size() / 2; ++i)
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
	public:
		vector<int> theArray;
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

	
	class HuffmanTree
	{
	private:
		struct HuffmanNode
		{
			int weight;
			string tag;
			HuffmanNode* left;
			HuffmanNode* right;
			HuffmanNode(int w = 0, string s = "", HuffmanNode* l = nullptr, HuffmanNode* r = nullptr) :
				weight(w), tag(s), left(l), right(r) {}
		};
		struct cmp
		{
			bool operator()(HuffmanNode*  p1,HuffmanNode*  p2)
			{
				return p1->weight > p2->weight;
			}
		};
		HuffmanNode *root;
		HuffmanNode* createHuffmanTree(const vector<pair<string, int> > &tagWeightPairVec)
		{
			priority_queue<HuffmanNode*, vector<HuffmanNode*>, cmp> pq;
			for (auto p : tagWeightPairVec)
				pq.push(new HuffmanNode(p.second, p.first, nullptr, nullptr));
			while (pq.size() > 1)
			{
				HuffmanNode* min1 = pq.top();
				pq.pop();
				HuffmanNode* min2 = pq.top();
				pq.pop();
				HuffmanNode* mergedNode = new HuffmanNode(min1->weight + min2->weight, "null", min1, min2);
				pq.push(mergedNode);
			}
			return pq.top();
		}
		void makeEmpty(HuffmanNode* &t)
		{
			if (t != nullptr)
			{
				makeEmpty(t->left);
				makeEmpty(t->right);
				delete t;
			}
			t = nullptr;
		}
		void traverse(HuffmanNode* t,vector<char>& sequence,ostream& os)
		{
			if (t != nullptr)
			{
				sequence.push_back('0');
				traverse(t->left,sequence,os);
				sequence.back() = '1';
				traverse(t->right, sequence,os);
				sequence.pop_back();
				if (t->left == nullptr&&t->right == nullptr)
				{
					os << t->tag << " "<<t->weight<<" ";
					for (auto ch : sequence)
						os << ch;
					os << "\n";
				}
			}
		}
	public:
		HuffmanTree() { root = nullptr; }
		HuffmanTree(const vector<pair<string, int> > &tagWeightPairVec)
		{
			root = createHuffmanTree(tagWeightPairVec);
		}
		HuffmanTree(const HuffmanTree&) = delete;
		~HuffmanTree()
		{
			makeEmpty(root);
		}
		ostream&  operator<<(ostream &os)
		{
			vector<char> seq;
			traverse(root, seq, os);
			return os;
		}
	};
}