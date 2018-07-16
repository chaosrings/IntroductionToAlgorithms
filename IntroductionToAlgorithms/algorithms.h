#pragma once
#include <vector>
#include <iostream>
#include "DataStructures.h"
#include <algorithm>
#include <map>
#include <math.h>
#include <queue>
#include <functional>
#include <Windows.h>
#include <time.h>
using namespace DataStructures;
using namespace std;
namespace SortMethods
{
	void insertionSort(vector<int> &array, int left, int right)
	{
		for (int i = left + 1; i <= right; i++)
		{
			int tmp = array[i];
			int j = i;
			for (; j>left&&array[j - 1]>tmp; j--)
			{
				array[j] = array[j - 1];
			}
			array[j] = tmp;
		}
	}
	int midian3(vector<int> & array, int left, int right)
	{
		int center = (left + right) / 2;
		if (array[center]<array[left])
			swap(array[center], array[left]);
		if (array[right]<array[left])
			swap(array[right], array[left]);
		if (array[right]<array[center])
			swap(array[center], array[right]);


		swap(array[center], array[right - 1]);
		return array[right - 1];
	}
	void merge(vector<int> &vec, int left, int mid, int right)
	{
		vector<int> temp(right-left+1, 0);
		int r = 0, r1 = left, r2 = mid + 1;
		for (; r1 <= mid&&r2 <=right; ++r)
		{
			if (vec[r1] < vec[r2])
				temp[r] = vec[r1++];
			else
				temp[r] = vec[r2++];
		}
		while (r1<=mid)
		{
			temp[r++] = vec[r1++];
		}
		while (r2<=right)
		{
			temp[r++] = vec[r2++];
		}
		for (int i = right; i >= left; --i)
			vec[i] = temp[--r];
	}
	void mergeSort(vector<int> &vec, int left, int right)
	{
		if (left < right)
		{
			int mid = (left + right) / 2;
			mergeSort(vec, left, mid);
			mergeSort(vec, mid + 1, right);
			merge(vec, left, mid, right);
		}
	}
	void quickSort(vector<int> &array, int left, int right)
	{
		if (left + 10<right)
		{
			int p = midian3(array, left, right);
			int i = left, j = right - 1;
			for (;;)
			{
				while (array[++i]<p) {};
				while (array[--j]>p) {};
				if (i<j)
					swap(array[i], array[j]);
				else
					break;
			}
			swap(array[i], array[right - 1]);
			quickSort(array, left, i - 1);
			quickSort(array, i + 1, right);
		}
		else
			insertionSort(array, left, right);
	}
	void countSort(vector<int>& A, int n, int k)
	{
		int *C = new int[k + 1];
	
		int *temp = new int[n];
		for (int i = 0; i <= k; ++i)
			C[i] = 0;
		for (int i = 0; i < n; ++i)
			++C[A[i]];
		for (int i = 1; i <= k; ++i)
			C[i] = C[i] + C[i - 1];
		for (int i = n - 1; i >= 0; --i)
		{
			temp[C[A[i]] - 1] = A[i];
			--C[A[i]];
		}
		for (int i = 0; i < n; ++i)
			A[i] = temp[i];
		delete[] C;
		delete[] temp;
	}
	void radixSort(vector<int>& A, int basicRadix,int maxIterTimes)
	{
		int n = A.size();
		vector<int> C(basicRadix + 1, 0);
		vector<int> B(n, 0);
		int E = 1;
		int iterCnt = 0;
		while (iterCnt<maxIterTimes)
		{
			for (int i = 0; i < n; ++i)
			{
				int radix = (A[i] % (E*basicRadix)) / E;
				++C[radix];
			}
			for (int i = 1; i <= basicRadix; ++i)
				C[i] = C[i] + C[i - 1];
			for (int i = n - 1; i >= 0; --i)
			{
				int radix = (A[i] % (E*basicRadix)) / E;
				B[C[radix] - 1] = A[i];
				--C[radix];
			}
			for (int i = 0; i < n; ++i)
				A[i] = B[i];
			C.assign(basicRadix + 1, 0);
			E *= basicRadix;
			++iterCnt;
		}
	}
	void BucketSort(vector<int>& A, int BucketNum,int maxNum)
	{
		ListNode* Buckets = new ListNode[BucketNum+1];
		int n = A.size();
		for (int i = 0; i < n; ++i)
		{
			int bucketIndex =static_cast<int>(floor((1.f*A[i] / maxNum)*BucketNum));
			if (Buckets[bucketIndex].next == nullptr)
				Buckets[bucketIndex].next = new ListNode(A[i], nullptr);
			else
			{
				ListNode* p = &Buckets[bucketIndex];
				while (p->next!=nullptr&&p->next->val<A[i])
				{
					p = p->next;
				}
				p->next = new ListNode(A[i], p->next);
			}
		}
		for (int i = 0, curPos=0;curPos<n&& i < BucketNum; ++i)
		{
			ListNode* iter = Buckets[i].next;
			while (iter!=nullptr)
			{
				ListNode* old = iter;
				A[curPos++] = iter->val;
				iter = iter->next;
				delete old;
			}

		}
		delete[]  Buckets;
	}
	void quickselect(vector<int>& array, int left, int right, int k)
	{
		if (left + 10<right)
		{
			int p = midian3(array, left, right);
			int i = left, j = right - 1;
			for (;;)
			{
				while (array[++i]<p) {};
				while (array[--j]>p) {};
				if (i<j)
					swap(array[i], array[j]);
				else
					break;
			}
			swap(array[i], array[right - 1]);
			if (k == i + 1)
				return;
			else if (k>i + 1)
				quickselect(array, i + 1, right, k);
			else
				quickselect(array, left, i - 1, k);
		}
		else
			insertionSort(array, left, right);
	}
};
namespace DivideAndConquer
{
	struct  Point
	{
		float x;
		float y;
		Point(float _x, float _y) :x(_x), y(_y) {};
		Point() {
			x = 0;
			y = 0;
		}
	};
	typedef  vector<vector<int>> Matrix;
	typedef  vector<Point>    Points;
	Matrix getSubMat(Matrix &mat, int TopLeftX, int TopLeftY, int BottomRightX, int BottomRightY)
	{
		Matrix ans;
		int ansRows = 0;
		for (int i = TopLeftX; i <= BottomRightX; ++i)
		{
			ans.push_back(vector<int>());
			ansRows++;
			for (int j = TopLeftY; j <= BottomRightY; ++j)
			{
				ans[ansRows-1].push_back(mat[i][j]);
			}
		}
		return ans;
	}
	Matrix MatSub(Matrix& mat1, Matrix& mat2)
	{
		Matrix ans;
		ans.assign(mat1.begin(), mat1.end());
		for (int i = 0; i < ans.size(); ++i)
		{
			for (int j = 0; j < ans[i].size(); ++j)
			{
				ans[i][j] -= mat2[i][j];
			}
		}
		return ans;
	}
	Matrix MatAdd(Matrix& mat1, Matrix& mat2)
	{
		Matrix ans;
		ans.assign(mat1.begin(), mat1.end());
		for (int i = 0; i < ans.size(); ++i)
		{
			for (int j = 0; j < ans[i].size(); ++j)
			{
				ans[i][j] += mat2[i][j];
			}
		}
		return ans;
	}
	Matrix unionPart(Matrix& part1, Matrix& part2, Matrix& part3, Matrix& part4)
	{
		Matrix ans(part1.size() + part3.size());
		for (int i = 0; i < part1.size(); ++i)
		{
			for (int j = 0; j < part1[i].size(); ++j)
			{
				ans[i].push_back(part1[i][j]);
			}
		}
		for (int i = 0; i < part2.size(); ++i)
		{
			for (int j = 0; j < part2[i].size(); ++j)
			{
				ans[i].push_back(part2[i][j]);
			}
		}
		for (int i = 0; i < part3.size(); ++i)
		{
			for (int j = 0; j < part3[i].size(); ++j)
			{
				ans[part1.size() + i].push_back(part3[i][j]);
			}
		}
		for (int i = 0; i < part4.size(); ++i)
		{
			for (int j = 0; j < part4[i].size(); ++j)
			{
				ans[part1.size() + i].push_back(part4[i][j]);
			}
		}
		return ans;
	}
	Matrix  MatMul(Matrix& mat1, Matrix& mat2)
	{
		if (mat1.size() == 1)
			return { {mat1[0][0] * mat2[0][0]} };
		else
		{
			int n = mat1.size();
			Matrix a = getSubMat(mat1, 0, 0, n / 2 - 1, n / 2 - 1);
			Matrix b = getSubMat(mat1, 0, n / 2, n / 2 - 1, n - 1);
			Matrix c = getSubMat(mat1, n / 2, 0, n -1, n/2 - 1);
			Matrix d = getSubMat(mat1, n / 2, n / 2, n - 1, n - 1);
			Matrix e = getSubMat(mat2, 0, 0, n / 2 - 1, n / 2 - 1);
			Matrix f = getSubMat(mat2, 0, n / 2, n / 2 - 1, n - 1);
			Matrix g = getSubMat(mat2, n / 2, 0, n-1, n/2 - 1);
			Matrix h = getSubMat(mat2, n / 2, n / 2, n - 1, n - 1);

			Matrix P1 = MatMul(a, MatSub(f, h));
			Matrix P2 = MatMul(MatAdd(a, b), h);
			Matrix P3 = MatMul(MatAdd(c, d), e);
			Matrix P4 = MatMul(d, MatSub(g, e));
			Matrix P5 = MatMul(MatAdd(a, d), MatAdd(e, h));
			Matrix P6 = MatMul(MatSub(b, d), MatAdd(g, h));
			Matrix P7 = MatMul(MatSub(a, c), MatAdd(e, f));

			Matrix part1 = MatAdd(P5, MatAdd(MatSub(P4, P2), P6));
			Matrix part2 = MatAdd(P1, P2);
			Matrix part3 = MatAdd(P3, P4);
			Matrix part4 = MatAdd(P1, MatSub(MatSub(P5, P3), P7));
			return unionPart(part1, part2, part3, part4);
		}
	}
	Matrix  MatMulBaoli(Matrix& mat1, Matrix& mat2)
	{
		int n = mat1.size();
		Matrix ans(n, vector<int>(n, 0));
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
				for (int k = 0; k < n; ++k)
				{
					ans[i][j] += mat1[i][k] * mat2[k][j];
				}
			}
		}
		return ans;
	}
	float distance(Point p1, Point p2)
	{
		return sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
	}
	bool cmpY(DivideAndConquer::Point & p1, DivideAndConquer::Point & p2)
	{
		return p1.y < p2.y;
	}
	float getNearestDist(Points & points,int left,int right)
	{
		if (left>=right)
			return 65535;
		else if (right-left==1)
			return distance(points[left], points[right]);
		else
		{
			int mid = (left + right) / 2;
			float delta = min(getNearestDist(points, left, mid), getNearestDist(points, mid + 1, right));
			Points DeltaNearPoints;

			int iterLeft = mid;
			while (iterLeft>=left&&points[iterLeft].x>points[mid].x-delta)
				DeltaNearPoints.push_back(points[iterLeft--]);
			int iterRight = mid + 1;
			while (iterRight<=right&&points[iterRight].x<points[mid].x+delta)
				DeltaNearPoints.push_back(points[iterRight++]);
			
			sort(DeltaNearPoints.begin(), DeltaNearPoints.end(), cmpY);
			float minDist = delta;
			for (int i = 0; i < DeltaNearPoints.size(); ++i)
			{
				for (int j = i + 1; j < DeltaNearPoints.size(); ++j)
				{
					if (DeltaNearPoints[j].y > DeltaNearPoints[i].y + delta)
						break;
					else
					{
						float curDist = distance(DeltaNearPoints[i], DeltaNearPoints[j]);
						if (curDist < minDist)
							minDist = curDist;
					}
				}
			}
			return minDist;
		}
	}
	float getNearestDistBaoli(Points &points)
	{
		float minDist = INT_MAX;
		for (int i = 0; i < points.size(); ++i)
		{
			for (int j = i + 1; j < points.size(); ++j)
			{
				float curDist = distance(points[i], points[j]);
				if (curDist < minDist)
					minDist = curDist;
			
			}
		}
		return minDist;
	}
};
namespace DynamicProgramming
{
	
	void program1()
	{
		vector<vector<int> > cost = { {2,7,9,3,40,8,3},{4,8,5,6,4,5,6} };
		vector<vector<int> > t = { {2,3,1,3},{2,1,2,2} };
		int n = cost[0].size() - 2;
		vector<vector<int> >  dp(2, vector<int>(n, 0));
		vector<int>		res(n,0);
		dp[0][0] = cost[0][0] + cost[0][1];
		dp[1][0] = cost[1][0] + cost[1][1];
		res[0] = cost[0][0] > cost[1][0];
		for (int i = 1; i < n; ++i)
		{
			dp[0][i] = min(dp[0][i - 1] + cost[0][i + 1], dp[1][i - 1] + t[1][i - 1] + cost[0][i + 1]);
			dp[1][i]= min(dp[1][i - 1] + cost[1][i + 1], dp[0][i - 1] + t[0][i - 1] + cost[1][i + 1]);
			if(i!=n-1)
				res[i] = dp[0][i] >= dp[1][i];
		}
		int minDist= min(dp[0].back() + cost[0].back(), dp[1].back() + cost[1].back());
		cout << minDist << endl;
		res[res.size() - 1] = dp[0].back() + cost[0].back() > dp[1].back() + cost[1].back();
		for (auto num : res)
		{
			cout << num << " ";
		}
		cout << endl;
	}
	void printLCS(const string &str1, const string &str2, const vector<vector<int> > & dp, int i, int j, string & ans)
	{
		if (i == -1 || j == -1)
			return ;
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
		vector<int> aline(str2.size() + 1, 0);
		vector<vector<int> > dp(str1.size() + 1, aline);
		for (unsigned int i = 1; i <= str1.size(); ++i)
		{
			for (unsigned int j = 1; j <= str2.size(); ++j)
			{
				dp[i][j] = str1[i] == str2[j] ? (dp[i - 1][j - 1] + 1) : max(dp[i - 1][j], dp[i][j - 1]);
			}
		}
		string ans = "";
		printLCS(str1, str2, dp, str1.size(), str2.size(), ans);
		return ans;
	}
	void getmatrixM(vector<int> & p)
	{
		p = { 30,35,15,5,10,20,25 };//对矩阵Mat(i) p[i]为行数,p[i+1]为列数
		int n = p.size() - 1;
		vector<vector<int> > m(n, vector<int>(n, INT_MAX));

		for (int i = 0; i < n; ++i)
		{
			m[i][i] = 0;
		}
		for (int l = 1; l <= n - 1; ++l)
		{
			for (int begin = 0; begin <= n - l - 1; ++begin)
			{
				int end = begin + l;
				for (int k = begin; k < end; ++k)
				{
					int times = m[begin][k] + m[k + 1][end] + p[begin] * p[k+1] * p[end+1];
					if (times < m[begin][end])
					{
						m[begin][end] = times;
					}
				}
			}
		}
		cout << m[0][5] << endl;
	}

};

namespace GraphicAlgorithms
{

	map<string,string> prim(const vector<string>& vertices, //顶点集合
		map<string,vector<pair<string,int>>> & edges  //一个map记录顶点连接的其他点和权值
			 )
	{
		map<string, string> parent;
		auto getMinVertex=[](const map<string, int> & key)
		{
			string ans = (*key.begin()).first;
			int curMin = (*key.begin()).second;
			for (auto entry : key)
			{
				if (entry.second < curMin)
				{
					ans = entry.first;
					curMin = entry.second;
				}
			}
			return ans;
		};
		
		map<string, int> key;
		for (int i = 0; i < vertices.size(); ++i)
			key[vertices[i]] = i == 0 ? 0 : INT_MAX;
		parent[vertices[0]] = "NULL";
		while (key.size()>0)
		{
			string curVertex = getMinVertex(key);
			key.erase(curVertex);
			auto adjacentVertices = edges[curVertex];
			for (auto vertex :adjacentVertices)
			{
				if (key.find(vertex.first) != key.end()   //连接点还在未被接受
					&& vertex.second < key[vertex.first])   //边的权要小于key中的值
				{
					parent[vertex.first] = curVertex;
					key[vertex.first] = vertex.second;
				}
			}
		}
		return parent;
	 }

	map<string,string> Kruskal(const vector<string>& vertices, //顶点集合
		map<string, vector<pair<string, int>>> & edges)
	{
		map<string, string> parent;
		priority_queue<Edge>  edgesQueue;
		for (auto iter = edges.begin(); iter != edges.end(); ++iter)
		{
			for (auto & adjacentVertex : (*iter).second)  //(*iter).second是连接此点的点集
			{
				if ((*iter).first<adjacentVertex.first )    //去除a-b,b-a添加两次的情况
				{
					edgesQueue.push(Edge((*iter).first, adjacentVertex.first, adjacentVertex.second));
				}
			}
		}
		map<string, int> indices;
		for (int i = 0; i < vertices.size(); ++i)
			indices[vertices[i]] = i;
		DisJointSet disjointSet(indices.size());
		int acceptedEdgesNum = 0;
		while (acceptedEdgesNum<vertices.size()-1)
		{
			Edge probEdge = edgesQueue.top();
			edgesQueue.pop();
			int root1 = disjointSet.find(indices[probEdge.adjacentVertex1]);
			int root2 = disjointSet.find(indices[probEdge.adjacentVertex2]);
			if (root1 != root2)
			{
				disjointSet.unionSet(root1, root2);
				parent[probEdge.adjacentVertex2] = probEdge.adjacentVertex1;
				++acceptedEdgesNum;
			}
		}
		return parent;
	}

	pair<map<string,string>,map<string, int> > Dijkstra(string origin,
		const vector<string>& vertices,//顶点集合
		map<string, vector<pair<string, int>>> & edges  //一个map记录顶点连接的其他点和权值
	)
	{
		map<string, string> parent;
		map<string, int> dist;
		map<string, int >  minDist;
		for (auto vertex : vertices)
			dist[vertex] = INT_MAX;
		dist[origin] = 0;
		auto getMinVertex = [](const map<string, int> & key)
		{
			string ans = (*key.begin()).first;
			int curMin = (*key.begin()).second;
			for (auto entry : key)
			{
				if (entry.second < curMin)
				{
					ans = entry.first;
					curMin = entry.second;
				}
			}
			return ans;
		};
		while (!dist.empty())
		{
			string cur = getMinVertex(dist);
			minDist.insert(make_pair(cur, dist[cur]));
			dist.erase(cur);
			for (auto adjacentVertex : edges[cur])
			{
				if (dist.find(adjacentVertex.first) != dist.end() &&
					minDist[cur] + adjacentVertex.second < dist[adjacentVertex.first])
				{	
					dist[adjacentVertex.first] = minDist[cur] + adjacentVertex.second;
					parent[adjacentVertex.first] = cur;
				}
			}
		}
		return make_pair(parent, minDist);
	}

	typedef vector<vector<int> > Matrix;
	pair<Matrix, Matrix> FLOYD_WARSHALL(Matrix & W)
	{
		int n = W.size();
		Matrix prevD;
		prevD.assign(W.begin(), W.end());
		Matrix curD(W.begin(), W.end());

		Matrix prevParent(n, vector<int>(n, INT_MAX));
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
				if (!(i == j || W[i][j] == INT_MAX))
					prevParent[i][j] = i;
			}
		}
		Matrix curParent(n, vector<int>(n, INT_MAX));

		for (int k = 0; k < n; ++k)
		{
			for (int i = 0; i < n; ++i)
			{
				for (int j = 0; j < n; ++j)
				{
					int cur = prevD[i][k] == INT_MAX || prevD[k][j] == INT_MAX ? INT_MAX : (prevD[i][k] + prevD[k][j]);
					if (prevD[i][j] <= cur)
					{
						curD[i][j] = prevD[i][j];
						curParent[i][j] = prevParent[i][j];
					}
					else
					{
						curD[i][j] = cur;
						curParent[i][j] = prevParent[k][j];
					}
				}
			}
			prevD.assign(curD.begin(), curD.end());
			prevParent.assign(curParent.begin(), curParent.end());
		}
		return make_pair(curParent, curD);
	}
	void print_all_pairs_shortest_path(vector<vector<int>> & parent, int i, int j)
	{
		if (i == j)
			cout << i + 1 << " ";
		else if (parent[i][j] == INT_MAX)
		{
			cout << "没有路径" << endl;
		}
		else
		{
			int k = parent[i][j];
			print_all_pairs_shortest_path(parent, i, k);
			cout << j + 1 << " ";
		}
	}
}
namespace TestMethod
{
	void printVec(vector<int> &A)
	{
		for (int i = 0; i < A.size(); ++i)
			cout << A[i] << " ";
		cout << endl;
	}
	void printMat(DivideAndConquer::Matrix & mat)
	{
		for (int i = 0; i < mat.size(); ++i)
		{
			for (int j = 0; j < mat[i].size(); ++j)
				cout << mat[i][j] << " ";
			cout << endl;
		}
	}
	bool cmpX(DivideAndConquer::Point & p1, DivideAndConquer::Point & p2)
	{
		return p1.x < p2.x;
	}

	void printMST(string curVertexTag, map<string, string> &parent)
	{
		queue<string> q;
		q.push(curVertexTag);
		q.push("NULL");
		while (!q.empty())
		{
			if (q.front() == "NULL")
			{
				if (q.size() == 1)
					break;
				cout << endl;
				q.pop();
				q.push("NULL");
				continue;
			}
			else
			{
				string curVertex = q.front();
				cout << curVertex << " ";
				q.pop();
				for (auto entry : parent)
				{
					if (entry.second == curVertex)
						q.push(entry.first);
				}
			}
		}
		cout << endl;
	}
	void TestSortMethods()
	{
		using namespace SortMethods;
		int scale = 10;
		vector<int> A(scale, 0);
		int numRange = 1000;
		int radixSortIterTimes = 4;
		while (scale <= 1000000)
		{
			cout << "在输入的数规模在" << scale << "数量级,数的范围在0~" << numRange << endl;
			A.assign(scale, 0);

			for (int i = 0; i <= 5; ++i)
			{

				srand(time(NULL));
				for (int i = 0; i < A.size(); i++)
				{
					A[i] = rand() % numRange;

				}
				//printVec(A);
				LARGE_INTEGER t1, t2, tc;
				QueryPerformanceFrequency(&tc);
				QueryPerformanceCounter(&t1);
				switch (i)
				{
				case 0:
					if (scale <= 10000)
					{
						insertionSort(A, 0, A.size() - 1);
						cout << "插入排序";
					}
					else
						cout << "速度太慢,不考虑" << endl;
					break;
				case 1:
					mergeSort(A, 0, A.size() - 1);
					cout << "归并排序";
					break;
				case 2:
					quickSort(A, 0, A.size() - 1);
					cout << "快速排序";
					break;
				case 3:
					countSort(A, A.size() - 1, numRange);
					cout << "计数排序";
					break;
				case 4:
					radixSort(A, 10, radixSortIterTimes);
					cout << "基数排序";
					break;
				case 5:
					BucketSort(A, scale / 10, numRange);
					cout << "桶排序";
					break;
				default:
					break;
				}
				QueryPerformanceCounter(&t2);
				//printVec(A);
				if (!(scale > 10000 && i == 0))
					cout << "花费时间" << (t2.QuadPart - t1.QuadPart)*1.0 / tc.QuadPart << endl;
			}
			scale *= 5;
		}
	}
	void TestDivideAndConquar()
	{
		DivideAndConquer::Matrix part1 = {
			{ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 },
			{ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 },
			{ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 },
			{ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 },
			{ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 },
			{ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 },
			{ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 },
			{ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 },
			{ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 },
			{ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 },
			{ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 },
			{ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 },
			{ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 },
			{ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 },
			{ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 },
			{ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 } };
		DivideAndConquer::Matrix part2 = part1;
		part2[0][0] = 0;
		DivideAndConquer::Matrix m = DivideAndConquer::MatMul(part1, part2);
		cout << "分治矩阵乘法" << endl;
		printMat(m);
		m = DivideAndConquer::MatMulBaoli(part1, part2);
		cout << "暴力矩阵乘法" << endl;
		printMat(m);

		DivideAndConquer::Points points;
		srand(time(NULL));
		int range = 100;
		for (int i = 0; i <100; ++i)
		{
			float _x = 1.f*(rand() % range) / range * 35;
			float _y = 1.f*(rand() % range) / range * 35;
			points.push_back(DivideAndConquer::Point(_x, _y));
		}
		sort(points.begin(), points.end(), cmpX);
		cout << "分治求解最近点" << DivideAndConquer::getNearestDist(points, 0, points.size() - 1) << endl;
		cout << "暴力求解最近点" << DivideAndConquer::getNearestDistBaoli(points) << endl;
	}


	void TestGraphicAlgorthims()
	{
		vector<string>  vertices = { "s","t","y","x","z" };
		map<string, vector<pair<string, int> > > edges;
		edges["s"].push_back(make_pair("t", 6));
		edges["s"].push_back(make_pair("y", 7));

		edges["t"].push_back(make_pair("x", 5));
		edges["t"].push_back(make_pair("y", 8));
		edges["t"].push_back(make_pair("z", -4));

		edges["x"].push_back(make_pair("t", -2));

		edges["z"].push_back(make_pair("x", 7));
		edges["z"].push_back(make_pair("s", 2));

		edges["y"].push_back(make_pair("x", -3));
		edges["y"].push_back(make_pair("z", 9));
		auto     ans = GraphicAlgorithms::Dijkstra("s", vertices, edges);
		for (int i = 0; i < vertices.size(); ++i)
			cout << "点" << vertices[i] << "到原点的距离为" << ans.second[vertices[i]] << endl;
		printMST("s", ans.first);
		cout << endl;

		vector<vector<int> >  W = {
			{ 0,3,8,INT_MAX,-4 },
			{ INT_MAX,0,INT_MAX,1,7 },
			{ INT_MAX,4,0,INT_MAX,INT_MAX },
			{ 2,INT_MAX,-5,0,INT_MAX },
			{ INT_MAX,INT_MAX,INT_MAX,6,0 }
		};
		auto secondAns = GraphicAlgorithms::FLOYD_WARSHALL(W);
		for (int i = 0; i < W.size(); ++i)
		{
			for (int j = i + 1; j < W[i].size(); ++j)
			{

				cout << "点对" << i + 1 << "," << j + 1 << " 路径长度为" << secondAns.second[i][j] << endl;
				GraphicAlgorithms::print_all_pairs_shortest_path(secondAns.first, i, j);
				cout << endl;

			}
		}
		/*for (auto entry : ans.first)
		{
		cout << entry.second << "连接" << entry.first << " 权重: ";
		for (auto vertex : edges[entry.second])
		{
		if (vertex.first == entry.first)
		{
		cout << vertex.second << endl;
		break;
		}
		}
		}*/
	}

	int partition(int A[], int p, int r)
	{
		int  x = A[r];
		int  i = p - 1;
		for (int j = p; j < r; ++j)
		{
			if (A[j] <= x)
			{
				++i;
				swap(A[i], A[j]);
			}
		}
		swap(A[i + 1], A[r]);
		return i + 1;
	}
	void Quicksort(int A[], int p, int r)
	{
		if (p < r)
		{
			int i = partition(A, p, r);
			Quicksort(A, p, i - 1);
			Quicksort(A, i + 1, r);
		}
	}

}