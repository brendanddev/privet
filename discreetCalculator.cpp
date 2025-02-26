
#include <iostream> 
#include <cmath> 

using namespace std; 

// Function for GCF (Greatest Common Factor)
int gcf(int a, int b) { 
    while (b != 0) { 
        int tempA = b; 
        b = a % b; 
        a = tempA;.
    }
    return a;
}

// Function for finding the LCM (lowest Common Multiplier)
int lcm(int a, int b) { 
    return (a / gcf(a, b)) * b;
}


int congruenceCalculator(int a, int b, int m) { 
    int remA = a % m; 
    int remB = b % m; 
    
    cout << "Remainder of " << a << ", (mod " << m << "), is: " << remA << endl;
    cout << "Remainder of " << b << ", (mod " << m << "), is: " << remB << endl;

    // Checks if the remainders are equal, meaning the two integers 'a' and 'b' are said to be congruent mod 'm'
    if (remA == remB) {
        cout << a << ", and " << b << " are congruent mod " << m << endl;
    // Not congruent mod 'm'
    } else {
        cout << a << ", and " << b << " are not congruent mod " << m << endl;
    }
}


// Function that determines if an integer is prime or not
bool primeCheck(int n) {
    if (n <= 1) {
        return false;
    }
    if (n == 2 || n == 3) {
        return true;
    }
    if (n % 2 == 0 || n % 3 == 0) {
        return false;
    }
    int sqrtLimit = sqrt(n);
    for (int i = 5; i <= sqrtLimit; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) {
            return false;
        }
    }
    return true;
}

// Checks if an intger is a perfect square
bool perfSquare(int x) { 
    int squareRoot = sqrt(x);
    return (squareRoot * squareRoot == x);
}

// Checks if an integer belongs to a fibonacci sequence
bool fibonacciCheck(int n) { 
    return perfSquare(5 * n * n + 4) || perfSquare(5 * n * n - 4);
}

// Main function acting as the entry poin of the code and Discreet Calculator program
int main() {
    int choice, a, b, m;

   
    do {
        cout << "\nBrendan's Discreet Calculator Math Menu:\n"; 
        cout << "1. Find GCF\n"; 
        cout << "2. Find LCM\n";
        cout << "3. Check Congruence\n";
        cout << "4. Prime Check\n";
        cout << "5. Fibonacci Check\n"; 
        cout << "6. Exit\n";
        cout << "Enter your choice, or '6' to quit";
        cin >> choice; 
        
        switch(choice) {
            case 1:
                cout << "Enter a value 'a': ";
                cin >> a;
                cout << "Enter a value 'b': ";
                cin >> b;
                cout << "The GCF is: " << gcf(a, b);
                break;
            case 2:
                cout << "Enter a value 'a': ";
                cin >> a;
                cout << "Enter a value 'b': ";
                cin >> b;
                cout << "The LCM of " << a << " and " << b << " is: " << lcm(a, b);
                break;
            case 3:
                cout << "Enter an integer for 'a': ";
                cin >> a;
                cout << "Enter an integer for 'b': ";
                cin >> b;
                cout << "Enter the modulus 'm': ";
                cin >> m;

                if (m != 0) {
                    congruenceCalculator(a, b, m);
                } else {
                    cout << "The mod cannot be 0!" << endl;
                }
                break;
            case 4:
                cout << "Enter an integer: ";
                cin >> a;
                if (primeCheck(a)) {
                    cout << a << ", is a prime number!\n";
                } else {
                    cout << a << ", is not a prime number!\n";
                }
                break;
            case 5:
                cout << "Enter an integer: ";
                cin >> a;

                if (fibonacciCheck(a)) {
                    cout << a << ", is a Fibonacci number!\n";
                } else {
                    cout << a << ", is not a Fibonacci number!\n";
                }
                break;
            case 6:
                cout << "Exiting the Calculator! Goodbye!";
                break;
            default:
                cout << "This was not a choice, please try again!";
                break;
        }
    } while (choice != 6);
    return 0;
}
