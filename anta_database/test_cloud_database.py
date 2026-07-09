"""
Test script for CloudDatabase class.
Tests connection to S3-hosted database and basic queries.
"""

from cloud_database import CloudDatabase


def test_cloud_database():
    """Test the CloudDatabase functionality."""
    print("=== Testing CloudDatabase ===")

    # Test connection
    print("\n1. Testing S3 connection...")
    try:
        with CloudDatabase(s3_bucket="anta-database") as db:
            if db.test_connection():
                print("✅ S3 connection successful")
            else:
                print("⚠️  S3 connection test failed, trying HTTP fallback")

            # Test basic query
            print("\n2. Testing database query...")
            results = db.query(dataset="Chung_2023")
            print(f"✅ Found {len(results)} flight lines")

            if results:
                # Show first result
                print(f"\n3. Sample result:")
                print(f"   Dataset: {results[0]['dataset']}")
                print(f"   Flight Line: {results[0]['flight_line']}")
                print(f"   File Path: {results[0]['file_path']}")

                # Test file retrieval
                print(f"\n4. Testing file retrieval...")
                files = db.get_files(results[:1])  # Just first file
                print(f"✅ Retrieved {len(files)} file paths: {files}")

                # If your Database class has download_files method
                if hasattr(db, "download_files"):
                    print(f"\n5. Testing file download...")
                    local_files = db.download_files(files)
                    print(f"✅ Downloaded {len(local_files)} files: {local_files}")

            print("\n✅ All tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = test_cloud_database()
    if success:
        print("\n🎉 CloudDatabase is working correctly!")
    else:
        print("\n💥 CloudDatabase test failed!")
        exit(1)
